import os

import gym
import gym_AVD
from gym.spaces.box import Box

from baselines import bench
from baselines.common.atari_wrappers import make_atari, wrap_deepmind

try:
    import pybullet_envs
    import roboschool
except ImportError:
    pass

bp = '/net/bvisionserver3/playpen10/ammirato/Data/HalvedRohitData/'
tp = '/net/bvisionserver3/playpen10/ammirato/Data/instance_detection_targets/uw_real_and_BB'

def make_env(env_id, seed, rank, log_dir):
    def _thunk():
        env = gym.make('AVD-v0')
        env.setup(AVD_path=bp, target_path=tp)
        #obs = env.reset()
        if log_dir is not None:
            env = bench.Monitor(env, os.path.join(log_dir, str(rank)))
        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        obs_shape = env.observation_space.shape
        #if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
        #    env = WrapPyTorch(env)
        return env

        #env = gym.make(env_id)
        #is_atari = hasattr(gym.envs, 'atari') and isinstance(env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        #if is_atari:
        #    env = make_atari(env_id)
        #env.seed(seed + rank)
        #if log_dir is not None:
        #    env = bench.Monitor(env, os.path.join(log_dir, str(rank)))
        #if is_atari:
        #    env = wrap_deepmind(env)
        ## If the input has shape (W,H,3), wrap for PyTorch convolutions
        #obs_shape = env.observation_space.shape
        #if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
        #    env = WrapPyTorch(env)
        #return env

    return _thunk


class WrapPyTorch(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(WrapPyTorch, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0,0,0],
            self.observation_space.high[0,0,0],
            [obs_shape[2], obs_shape[1], obs_shape[0]]
        )

#    def _observation(self, observation):
#        return observation.transpose(2, 0, 1)

    #phil add
    def observation(self, observation):
        return observation
