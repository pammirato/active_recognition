import torch
import torch.nn as nn
import torch.nn.functional as F
from distributions import Categorical, DiagGaussian
from utils import orthogonal
from target_driven_instance_detection.model_defs.TDID import TDID
import target_driven_instance_detection.utils as tdid_utils



class FFPolicy(nn.Module):
    def __init__(self):
        super(FFPolicy, self).__init__()

    def forward(self, inputs, states, masks):
        raise NotImplementedError

    def act(self, scene_imgs, target_imgs, states, masks, deterministic=False):
        value, x, states = self(scene_imgs,target_imgs, states, masks)
        action = self.dist.sample(x, deterministic=deterministic)
        action_log_probs, dist_entropy = self.dist.logprobs_and_entropy(x, action)
        return value, action, action_log_probs, states

    def evaluate_actions(self, scene_imgs, target_imgs, states, masks, actions):
        value, x, states = self(scene_imgs,target_imgs, states, masks)
        action_log_probs, dist_entropy = self.dist.logprobs_and_entropy(x, actions)
        return value, action_log_probs, dist_entropy, states


class CNNPolicy(FFPolicy):
    def __init__(self, num_inputs, action_space, use_gru, tdid_cfg):
        super(CNNPolicy, self).__init__()
        #self.conv1 = nn.Conv2d(num_inputs, 32, 8, stride=4)
        #self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        #self.conv3 = nn.Conv2d(64, 32, 3, stride=1)

        self.tdid = TDID(tdid_cfg)
        #self.tdid = self.tdid.cuda()
        self.tdid_out_shape = (int(num_inputs[0]/self.tdid._feat_stride),
                               int(num_inputs[1]/self.tdid._feat_stride))


        #self.pool1 = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(self.tdid.num_feature_channels, 128, 3, padding=2)
        self.pool2 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(128, 32, 3, padding=2)


        #self.num_features = 32 * int(self.tdid_out_shape[0]/4) * int(self.tdid_out_shape[1]/4)
        #self.linear1 = nn.Linear(self.num_features,512)
        self.linear1 = nn.Linear(32*7*10,512)
        #self.linear1 = nn.Linear(32*7*7, 512)

        #if use_gru:
        #    self.gru = nn.GRUCell(512, 512)

        self.critic_linear = nn.Linear(512, 1)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(512, num_outputs)
        #elif action_space.__class__.__name__ == "Box":
        #    num_outputs = action_space.shape[0]
        #    self.dist = DiagGaussian(512, num_outputs)
        else:
            raise NotImplementedError

        self.train()
        self.tdid.eval()
        self.reset_parameters()

    @property
    def state_size(self):
        if hasattr(self, 'gru'):
            return 512
        else:
            return 1

    def reset_parameters(self):
        #self.apply(weights_init)
        self.weights_init()

        relu_gain = nn.init.calculate_gain('relu')
        #self.conv1.weight.data.mul_(relu_gain)
        #self.conv2.weight.data.mul_(relu_gain)
        #self.conv3.weight.data.mul_(relu_gain)
        self.linear1.weight.data.mul_(relu_gain)

        #if hasattr(self, 'gru'):
        #    orthogonal(self.gru.weight_ih.data)
        #    orthogonal(self.gru.weight_hh.data)
        #    self.gru.bias_ih.data.fill_(0)
        #    self.gru.bias_hh.data.fill_(0)

        if self.dist.__class__.__name__ == "DiagGaussian":
            self.dist.fc_mean.weight.data.mul_(0.01)

    def forward(self, scene_data,target_data, states, masks):
        img_info = scene_data.shape[1:]
        scene_data = scene_data.permute(0,3,1,2)
        target_data = target_data.permute(0,3,1,2)
        #x = self.conv1(inputs / 255.0)
        #x = F.relu(x)

        #x = self.conv2(x)
        #x = F.relu(x)

        #x = self.conv3(x)
        #x = F.relu(x)
        _,_,x = self.tdid(target_data,scene_data,img_info)


        F.relu(x)
        #x = self.pool1(x)
        x = self.conv1(x)
        F.relu(x)
        x = self.pool2(x)
        x = self.conv2(x)



        #x = x.view(-1, self.tdid.num_feature_channels * int(self.tdid_out_shape[0]/4) * int(self.tdid_out_shape[1]/4))
        x = x.view(-1, 32*7*10)
        x = self.linear1(x)
        x = F.relu(x)

        #if hasattr(self, 'gru'):
        #    if inputs.size(0) == states.size(0):
        #        x = states = self.gru(x, states * masks)
        #    else:
        #        x = x.view(-1, states.size(0), x.size(1))
        #        masks = masks.view(-1, states.size(0), 1)
        #        outputs = []
        #        for i in range(x.size(0)):
        #            hx = states = self.gru(x[i], states * masks[i])
        #            outputs.append(hx)
        #        x = torch.cat(outputs, 0)
        return self.critic_linear(x), x, states



    def weights_init(self):
       
        tdid_utils.weights_normal_init(self.tdid, dev=0.01)
        #if cfg.USE_PRETRAINED_WEIGHTS:
        #    net.features = load_pretrained_weights(cfg.FEATURE_NET_NAME) 

        orthogonal(self.conv1.weight.data)
        if self.conv1.bias is not None:
            self.conv1.bias.data.fill_(0)

        orthogonal(self.conv2.weight.data)
        if self.conv2.bias is not None:
            self.conv2.bias.data.fill_(0)

        orthogonal(self.linear1.weight.data)
        if self.linear1.bias is not None:
            self.linear1.bias.data.fill_(0)

        orthogonal(self.critic_linear.weight.data)
        if self.critic_linear.bias is not None:
            self.critic_linear.bias.data.fill_(0)





def weights_init_mlp(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class MLPPolicy(FFPolicy):
    def __init__(self, num_inputs, action_space):
        super(MLPPolicy, self).__init__()

        self.action_space = action_space

        self.a_fc1 = nn.Linear(num_inputs, 64)
        self.a_fc2 = nn.Linear(64, 64)

        self.v_fc1 = nn.Linear(num_inputs, 64)
        self.v_fc2 = nn.Linear(64, 64)
        self.v_fc3 = nn.Linear(64, 1)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(64, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(64, num_outputs)
        else:
            raise NotImplementedError

        self.train()
        self.reset_parameters()

    @property
    def state_size(self):
        return 1

    def reset_parameters(self):
        self.apply(weights_init_mlp)

        """
        tanh_gain = nn.init.calculate_gain('tanh')
        self.a_fc1.weight.data.mul_(tanh_gain)
        self.a_fc2.weight.data.mul_(tanh_gain)
        self.v_fc1.weight.data.mul_(tanh_gain)
        self.v_fc2.weight.data.mul_(tanh_gain)
        """

        if self.dist.__class__.__name__ == "DiagGaussian":
            self.dist.fc_mean.weight.data.mul_(0.01)

    def forward(self, inputs, states, masks):
        x = self.v_fc1(inputs)
        x = F.tanh(x)

        x = self.v_fc2(x)
        x = F.tanh(x)

        x = self.v_fc3(x)
        value = x

        x = self.a_fc1(inputs)
        x = F.tanh(x)

        x = self.a_fc2(x)
        x = F.tanh(x)

        return value, x, states
