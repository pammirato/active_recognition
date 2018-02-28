import copy
import glob
import os
import time
import importlib

import gym
import gym_AVD
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from arguments import get_args
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
from envs_tdid import make_env
from kfac import KFACOptimizer
from model_tdid import CNNPolicy, MLPPolicy
from storage_tdid import RolloutStorage
from visualize import visdom_plot

#from target_driven_instance_detection.model_defs.TDID import TDID

tdid_cfg_file = 'configTEST' #NO FILE EXTENSTION!
tdid_cfg = importlib.import_module('target_driven_instance_detection.configs.'+tdid_cfg_file)
tdid_cfg = tdid_cfg.get_config()

args = get_args()

assert args.algo in ['a2c', 'ppo', 'acktr']
if args.recurrent_policy:
    assert args.algo in ['a2c', 'ppo'], \
        'Recurrent policy is not implemented for ACKTR'

num_updates = int(args.num_frames) // args.num_steps // args.num_processes

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

try:
    os.makedirs(args.log_dir)
except OSError:
    files = glob.glob(os.path.join(args.log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)


def main():
    print("#######")
    print("WARNING: All rewards are clipped or normalized so you need to use a monitor (see envs.py) or visdom plot to get true rewards")
    print("#######")

    os.environ['OMP_NUM_THREADS'] = '1'

    if args.vis:
        from visdom import Visdom
        viz = Visdom(port=args.port)
        win = None

    envs = [make_env(args.env_name, args.seed, i, args.log_dir)
                for i in range(args.num_processes)]

    #if args.num_processes > 1:
    envs = SubprocVecEnv(envs)
    #else:
    #    envs = DummyVecEnv(envs)

    #if len(envs.observation_space.shape) == 1:
    #    envs = VecNormalize(envs)

    obs_shape = envs.observation_space.spaces['scene_image'].shape
    target_shape = envs.observation_space.spaces['target_image'].shape
    #obs_shape = envs.observation_space.shape
    #obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])

    #if len(envs.observation_space.shape) == 3:
        #actor_critic = CNNPolicy(obs_shape[0], envs.action_space, args.recurrent_policy)
    actor_critic = CNNPolicy(obs_shape, envs.action_space, args.recurrent_policy, tdid_cfg)
    #else:
    #    assert not args.recurrent_policy, \
    #        "Recurrent policy is not implemented for the MLP controller"
    #    actor_critic = MLPPolicy(obs_shape[0], envs.action_space)

    if envs.action_space.__class__.__name__ == "Discrete":
        action_shape = 1
    else:
        action_shape = envs.action_space.shape[0]

    if args.cuda:
        actor_critic.cuda()

    if args.algo == 'a2c':
        optimizer = optim.RMSprop(actor_critic.parameters(), args.lr, eps=args.eps, alpha=args.alpha)
    elif args.algo == 'ppo':
        optimizer = optim.Adam(actor_critic.parameters(), args.lr, eps=args.eps)
    elif args.algo == 'acktr':
        optimizer = KFACOptimizer(actor_critic)

    rollouts = RolloutStorage(args.num_steps, args.num_processes, obs_shape,target_shape, envs.action_space, actor_critic.state_size)
    current_obs = torch.zeros(args.num_processes, *obs_shape)
    current_target = torch.zeros(2*args.num_processes, *target_shape)


    def update_current_obs(obs, target):
        #shape_dim0 = envs.observation_space.shape[0]
        #obs = torch.from_numpy(obs).float()
        #if args.num_stack > 1:
        #    current_obs[:, :-shape_dim0] = current_obs[:, shape_dim0:]
        #current_obs[:, -shape_dim0:] = obs

        num_chnls = obs.shape[-1]
        if args.num_stack > 1:
            current_obs[:,:,:, :-num_chnls] = current_obs[:,:,:, num_chnls:]
            current_target[:,:,:, :-num_chnls] = current_target[:,:,:, num_chnls:]
        current_obs[:,:,:, -num_chnls:] = torch.FloatTensor(obs)
        current_target[:,:,:, -num_chnls:] = torch.FloatTensor(target)
        

    obs = envs.reset()
    all_scene_images = []
    all_target_images = []
    for obs_ind in range(0,len(obs)):
        all_scene_images.append(obs[obs_ind]['scene_image'])
        all_target_images.append(obs[obs_ind]['target_image'])
    all_scene_images = np.stack(all_scene_images)
    #all_target_images = np.stack(all_target_images)
    all_target_images = torch.FloatTensor(match_and_concat_images_list(all_target_images))

    update_current_obs(all_scene_images, all_target_images)

    rollouts.observations[0].copy_(current_obs)
    rollouts.targets[0].copy_(current_target)

    # These variables are used to compute average rewards for all processes.
    episode_rewards = torch.zeros([args.num_processes, 1])
    final_rewards = torch.zeros([args.num_processes, 1])

    if args.cuda:
        current_obs = current_obs.cuda()
        current_target = current_target.cuda()
        #all_target_images = all_target_images.cuda()
        rollouts.cuda()

    start = time.time()
    for j in range(num_updates):
        for step in range(args.num_steps):
            print('STEP: {}'.format(step))
            # Sample actions
            value, action, action_log_prob, states = \
                actor_critic.act(Variable(rollouts.observations[step], volatile=True),
                                 Variable(rollouts.targets[step], volatile=True),
                #                 Variable(all_target_images, volatile=True),
                                 Variable(rollouts.states[step], volatile=True),
                                 Variable(rollouts.masks[step], volatile=True))
            cpu_actions = action.data.squeeze(1).cpu().numpy()

            # Obser reward and next obs
            obs, reward, done, info = envs.step(cpu_actions)
            reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
            episode_rewards += reward

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            final_rewards *= masks
            final_rewards += (1 - masks) * episode_rewards
            episode_rewards *= masks

            if args.cuda:
                masks = masks.cuda()

            if current_obs.dim() == 4:
                current_obs *= masks.unsqueeze(2).unsqueeze(2)
                #current_target *= np.concat([masks.unsqueeze(2).unsqueeze(2),masks.unsqueeze(2).unsqueeze(2)],axis=0)
            else:
                current_obs *= masks
                #current_target *= np.concat([masks,masks],axis=0)



            all_scene_images = []
            all_target_images = []
            for obs_ind in range(0,len(obs)):
                all_scene_images.append(obs[obs_ind]['scene_image'])
                all_target_images.append(obs[obs_ind]['target_image'])
            all_scene_images = np.stack(all_scene_images)
            #all_target_images = np.stack(all_target_images)
            all_target_images = torch.FloatTensor(match_and_concat_images_list(all_target_images))
            update_current_obs(all_scene_images,all_target_images)
            rollouts.insert(step, current_obs, current_target, states.data, action.data, action_log_prob.data, value.data, reward, masks)

        next_value = actor_critic(Variable(rollouts.observations[-1], volatile=True),
                                  Variable(rollouts.targets[-1], volatile=True),
              #                    Variable(all_target_images, volatile=True),
                                  Variable(rollouts.states[-1], volatile=True),
                                  Variable(rollouts.masks[-1], volatile=True))[0].data

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

        if args.algo in ['a2c', 'acktr']:

            #expanded_target_images = [] 
            #for process in range(args.num_processes):
            #    for step in range(args.num_steps):
            #        expanded_target_images.append(all_target_images[2*process:2*(process+1),:].cpu().numpy())
            #expanded_target_images = torch.FloatTensor(match_and_concat_images_list(expanded_target_images)).cuda()

            values, action_log_probs, dist_entropy, states = \
                 actor_critic.evaluate_actions(Variable(rollouts.observations[:-1].view(-1, *obs_shape)),
                                               Variable(rollouts.targets[:-1].view(-1, *target_shape)),
             #                                  Variable(expanded_target_images),
                                               Variable(rollouts.states[0].view(-1, actor_critic.state_size)),
                                               Variable(rollouts.masks[:-1].view(-1, 1)),
                                               Variable(rollouts.actions.view(-1, action_shape)))

            values = values.view(args.num_steps, args.num_processes, 1)
            action_log_probs = action_log_probs.view(args.num_steps, args.num_processes, 1)

            advantages = Variable(rollouts.returns[:-1]) - values
            value_loss = advantages.pow(2).mean()

            action_loss = -(Variable(advantages.data) * action_log_probs).mean()

            if args.algo == 'acktr' and optimizer.steps % optimizer.Ts == 0:
                # Sampled fisher, see Martens 2014
                actor_critic.zero_grad()
                pg_fisher_loss = -action_log_probs.mean()

                value_noise = Variable(torch.randn(values.size()))
                if args.cuda:
                    value_noise = value_noise.cuda()

                sample_values = values + value_noise
                vf_fisher_loss = -(values - Variable(sample_values.data)).pow(2).mean()

                fisher_loss = pg_fisher_loss + vf_fisher_loss
                optimizer.acc_stats = True
                fisher_loss.backward(retain_graph=True)
                optimizer.acc_stats = False

            optimizer.zero_grad()
            (value_loss * args.value_loss_coef + action_loss - dist_entropy * args.entropy_coef).backward()

            if args.algo == 'a2c':
                nn.utils.clip_grad_norm(actor_critic.parameters(), args.max_grad_norm)

            optimizer.step()
        elif args.algo == 'ppo':
            advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

            for e in range(args.ppo_epoch):
                if args.recurrent_policy:
                    data_generator = rollouts.recurrent_generator(advantages,
                                                            args.num_mini_batch)
                else:
                    data_generator = rollouts.feed_forward_generator(advantages,
                                                            args.num_mini_batch)

                for sample in data_generator:
                    observations_batch, states_batch, actions_batch, \
                       return_batch, masks_batch, old_action_log_probs_batch, \
                            adv_targ = sample

                    # Reshape to do in a single forward pass for all steps
                    values, action_log_probs, dist_entropy, states = actor_critic.evaluate_actions(Variable(observations_batch),
                                                                                                   Variable(states_batch),
                                                                                                   Variable(masks_batch),
                                                                                                   Variable(actions_batch))

                    adv_targ = Variable(adv_targ)
                    ratio = torch.exp(action_log_probs - Variable(old_action_log_probs_batch))
                    surr1 = ratio * adv_targ
                    surr2 = torch.clamp(ratio, 1.0 - args.clip_param, 1.0 + args.clip_param) * adv_targ
                    action_loss = -torch.min(surr1, surr2).mean() # PPO's pessimistic surrogate (L^CLIP)

                    value_loss = (Variable(return_batch) - values).pow(2).mean()

                    optimizer.zero_grad()
                    (value_loss + action_loss - dist_entropy * args.entropy_coef).backward()
                    nn.utils.clip_grad_norm(actor_critic.parameters(), args.max_grad_norm)
                    optimizer.step()

        rollouts.after_update()

        if j % args.save_interval == 0 and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            # A really ugly way to save a model to CPU
            save_model = actor_critic
            if args.cuda:
                save_model = copy.deepcopy(actor_critic).cpu()

            save_model = [save_model,
                            hasattr(envs, 'ob_rms') and envs.ob_rms or None]

            torch.save(save_model, os.path.join(save_path, args.env_name + ".pt"))

        if j % args.log_interval == 0:
            end = time.time()
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            print("Updates {}, num timesteps {}, FPS {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, entropy {:.5f}, value loss {:.5f}, policy loss {:.5f}".
                format(j, total_num_steps,
                       int(total_num_steps / (end - start)),
                       final_rewards.mean(),
                       final_rewards.median(),
                       final_rewards.min(),
                       final_rewards.max(), dist_entropy.data[0],
                       value_loss.data[0], action_loss.data[0]))
        if args.vis and j % args.vis_interval == 0:
            try:
                # Sometimes monitor doesn't properly flush the outputs
                win = visdom_plot(viz, win, args.log_dir, args.env_name, args.algo)
            except IOError:
                pass




def match_and_concat_images_list(img_list, min_size=None):
    """ 
    Stacks image in a list into a single ndarray 

    Input parameters:
        img_list: (list) list of ndarrays, images to be stacked. If images
                  are not the same shape, zero padding will be used to make
                  them the same size. 

        min_size (optional): (int) If not None, ensures images are at least
                             min_size x min_size. Default: None 

    Returns:
        (ndarray) a single ndarray with first dimension equal to the 
        number of elements in the inputted img_list    
    """
    #find size all images will be
    max_rows = 0 
    max_cols = 0 
    for img in img_list:
        max_rows = max(img.shape[1], max_rows)
        max_cols = max(img.shape[2], max_cols)
    if min_size is not None:
        max_rows = max(max_rows,min_size)
        max_cols = max(max_cols,min_size)

    #resize and stack the images
    for il,img in enumerate(img_list):
        resized_img = np.zeros((img.shape[0],max_rows,max_cols,img.shape[3]))
        resized_img[:,0:img.shape[1],0:img.shape[2],:] = img 
        img_list[il] = resized_img
    return np.concatenate(img_list,axis=0)












if __name__ == "__main__":
    main()
