from __future__ import division
from __future__ import print_function
import retro
import copy
from random import sample, randint, random, betavariate, uniform, seed
import itertools as it
import numpy as np
import skimage.color, skimage.transform
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init, Parameter
from torchvision import datasets, transforms
from torch.autograd import Variable
import math
from PIL import Image
import matplotlib.pyplot as plt
from collections import deque
from time import sleep
from datetime import datetime
import torch.multiprocessing as mp
import inspect
import gym
from copy import copy

class MarioXReward(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.lives = 2
        self.timeout = 300
        self.score = 0

    def reset(self):
        ob = self.env.reset()
        return ob

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        
        if reward <= 0:
            self.timeout -= 1
        else:
            self.timeout += 1
        if self.timeout == 0:
            done = True
            self.timeout = 300
            
        if info['lives'] < self.lives:
            done = True
            reward = -10
            self.timeout = 300
        self.lives = info['lives']
            
        self.timeout = min(self.timeout, 4800)
        reward = reward*info['speed']
        return ob, reward, done, info
        
torch.set_num_threads(1)
        
class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, dict):
        self.dict = dict

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.dict['mean']
        self.dict['count'] = min(self.dict['count'] + 1, 100000000)
        self.dict['mean'] += delta/self.dict['count']
        self.dict['var'] += (delta**2-self.dict['var'])/self.dict['count']
    
class CMAConv2d(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=5, stride=1, padding=2, bias=True, std_init=None, lr=1e-2, denom=200.0, maxlen=10):
        super(CMAConv2d, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.include_bias = bias
        self.stride = stride
        self.padding = padding
        if type(kernel_size) is int:
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.register_buffer('best_score', torch.tensor([1.0,1.0,1.0,1.0,1.0,1.0,1.0]))
        self.register_buffer('lifetime_score', torch.tensor([1.0,1.0,1.0,1.0,1.0,1.0,1.0]))
        self.register_buffer('mean_score', torch.tensor([0.0,0.0,0.0,0.0,0.0,0.0,0.0]))
        self.register_buffer('var_score', torch.tensor([1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6]))
        self.register_buffer('update_count', torch.tensor([1.0,1.0,1.0,1.0,1.0,1.0,1.0]))
        self.register_buffer('best_count', torch.tensor(1.0))
        self.register_buffer('best_mean', torch.tensor([0.0,0.0,0.0,0.0,0.0,0.0,0.0]))
        self.len_count = 10
        self.maxlen = maxlen
        self.denom = denom
        self.lr = lr
        self.weight_shape = [out_features, in_features, kernel_size[0], kernel_size[1]]
        if self.include_bias:
            self.param = Parameter(torch.Tensor(out_features*in_features*kernel_size[0]*kernel_size[1]+out_features).fill_(0))
            self.register_buffer('param_epsilon', torch.Tensor(out_features*in_features*kernel_size[0]*kernel_size[1]+out_features).fill_(0))
            self.register_buffer('param_cholesky', torch.Tensor(out_features*in_features*kernel_size[0]*kernel_size[1]+out_features).fill_(0))
        else:
            self.param = Parameter(torch.Tensor(out_features*in_features*kernel_size[0]*kernel_size[1]).fill_(0))
            self.register_buffer('param_epsilon', torch.Tensor(out_features*in_features*kernel_size[0]*kernel_size[1]).fill_(0))
            self.register_buffer('param_cholesky', torch.Tensor(out_features*in_features*kernel_size[0]*kernel_size[1]).fill_(0))
        self.register_buffer('param_covar',torch.eye(self.param.shape[0], self.param.shape[0]))
        self.reset_parameters()
        self.resample()
        for param in self.parameters():
            param.share_memory_()

    def reset_parameters(self):
        self.best_score.fill_(1)
        self.lifetime_score.fill_(1)
        self.mean_score.fill_(0)
        self.var_score.fill_(0)
        self.update_count.fill_(1)
        self.len_count = 10
        nn.init.eye_(self.param_covar)
        self.param.data.fill_(0)
        self.param_covar.data.div_(5*self.out_features**0.5)

    def score_update(self,score,target):
        if score > target.lifetime_score.data.item():
            target.lifetime_score.data.fill_(score)
        
    def CMA_update(self, score, target, thread):
        if score > target.lifetime_score[thread].data.item():
            target.lifetime_score[thread].data.fill_(score)

        delta = score-target.mean_score[thread]
        ratio = (delta)/(target.lifetime_score[thread]-target.mean_score[thread])
        ratio = ratio*2-1
        target.mean_score[thread] += delta/target.update_count[thread].item()
        target.var_score[thread].data.mul_(1-1/target.update_count[thread].item()).add_(1/target.update_count[thread].item(), delta**2)
        restart = False
        if ratio > 0:
            diff = target.param.data-(self.param.data+self.param_epsilon.data)
            target.param_covar.data.add_(ratio*0.05, diff.view(1,-1)*diff.view(-1,1)-target.param_covar.data)
            target.param.data.add_(ratio*0.05, self.param.data+self.param_epsilon.data-target.param.data)
        target.update_count[thread].add_(1).clamp_(0,target.denom)
        
        return restart

    def resample(self):
        outshape = list(self.param.shape)
        cholesky = torch.cholesky(self.param_covar.data.double().add(1e-4, torch.eye(self.param_covar.shape[0],self.param_covar.shape[1]).double())).float()
        self.param_epsilon.data = torch.randn_like(self.param).mul(cholesky.data).sum(1).view(outshape)#.mul(self.weight_sigma.data)
            
    def forward(self, input):
        if self.include_bias:
            bias = self.param[-self.weight_shape[0]:]
            weight = self.param[:-self.weight_shape[0]].view(self.weight_shape)
            if self.training:
                bias_epsilon = self.param_epsilon[-self.weight_shape[0]:]
                weight_epsilon = self.param_epsilon[:-self.weight_shape[0]].view(self.weight_shape)
                return F.conv2d(input,
                                weight + weight_epsilon,
                                bias=bias + bias_epsilon,
                                stride=self.stride, padding=self.padding)
            else:
                return F.conv2d(input, weight, bias = bias, stride=self.stride, padding=self.padding)
        else:
            weight = self.param.view(self.weight_shape)
            if self.training:
                weight_epsilon = self.param_epsilon.view(self.weight_shape)
                return F.conv2d(input,
                                weight + weight_epsilon,
                                stride=self.stride, padding=self.padding)
            else:
                return F.conv2d(input, weight, stride=self.stride, padding=self.padding)
    
class acNet(nn.Module):
    def __init__(self, n_actions, hidden_size=48, basis = 16):
        super(acNet, self).__init__()
        self.n_actions = n_actions
        self.best_score = nn.Parameter(torch.tensor(1.0))
        self.lifetime_best = 1.0
        self.prev_score = 1
        self.thread = nn.Parameter(torch.tensor(1.0))
        self.hidden_size = hidden_size
        self.sobelkern = torch.tensor([[[[1.,2.,1.,],[0.,0.,0.,],[-1.,-2.,-1.,]]]])
        self.sobelkern.div_(self.sobelkern.abs().sum())
        self.sobelkern = torch.cat((self.sobelkern, self.sobelkern.transpose(2,3)),0).unsqueeze(0)
        self.conv1 = CMAConv2d(34, hidden_size, 1, 1, 0)
        self.fc4 = CMAConv2d(self.hidden_size, self.n_actions*2, 1, 1, 0)
        linX = torch.linspace(-1,1,240//4)
        linY = torch.linspace(1,-1,224//4)
        self.vector = torch.Tensor(1,4,240//4,224//4)
        self.vector[0][0] = linY.expand(240//4,-1)
        self.vector[0][1] = linX.expand(224//4,-1).transpose(0,1)
        self.vector[0][2] = torch.atan2(self.vector[0][0], self.vector[0][1]).sin()
        self.vector[0][3] = torch.atan2(self.vector[0][0], self.vector[0][1]).cos()
        self.vector.add_(-self.vector.mean()).div_(self.vector.std())
        self.hidden = None
        
    def sample_noise(self):
        self.conv1.resample()
        self.fc4.resample()
        self.hidden = None
        
    def CMA_update(self, score, target, thread):
        _ = self.conv1.CMA_update(score, target.conv1, thread)
        _ = self.fc4.CMA_update(score, target.fc4, thread)
        
    def wiggleswish(self,U):
        return U.tanh()+U.relu()
        
    def forward(self, h, prev_actions):
        with torch.no_grad():
            h = h.div(255).add(-0.5)
            prev_actions = prev_actions.view(h.shape[0],-1,1,1).expand(-1,-1,h.shape[2],h.shape[3])
            kern = self.sobelkern.expand(3,-1,-1,-1,-1).contiguous().view(-1,1,3,3)
            j = F.conv2d(h, kern, groups=3, padding=1)
            j = torch.cat((j,F.conv2d(h, kern, groups=3, padding=2, dilation=2)),1)
            j = torch.cat((j,F.conv2d(h, kern, groups=3, padding=3, dilation=3)),1)
            h = torch.cat((h, self.vector, j, prev_actions), 1)
            h = self.conv1(h).relu()
            h = self.fc4(h).relu().mean((2,3)).view(h.shape[0],-1,2,self.n_actions)
            act = h[:,:,0,:]-h[:,:,1,:]
        return act
                    
class train_window(mp.Process):
    def __init__(self, net, lock, sticky):
        super(train_window, self).__init__()
        self.net = net
        self.sticky = sticky
        self.lock = lock
        
    def run(self):
        torch.manual_seed(randint(0,65535)+self.sticky)
        x = torch.randint(65540, (1,)).item()
        seed(x+self.sticky)
        levelstring = 'Level'+str(self.sticky+1)+'-1.state'
        game = retro.make(game='SuperMarioBros-Nes', state=levelstring)
        rewardwrapper = MarioXReward(game)
        s1 = rewardwrapper.reset()
        s1 = s1.reshape([1,s1.shape[0],s1.shape[1],3])
        s1 = torch.from_numpy(s1).float()
        s1 = s1.transpose(1,3)
        s1 = F.interpolate(s1, scale_factor=0.25, mode='bilinear', align_corners=True)
        done = False
        n = 9
        net = acNet(n)
        net.load_state_dict(self.net.state_dict())
        net.thread.data.fill_(self.sticky)
        net.sample_noise()
        ep_score = 0
        best_score = 0
        checkpoint_counter = 0
        first = True
        poop = torch.zeros(1,n)
        while True:
            with torch.no_grad():
                if done:
                    self.lock.acquire()
                    net.CMA_update(ep_score, self.net, self.sticky)
                    if self.sticky == 0 and checkpoint_counter >= 50:
                        print('saving model checkpoint..')
                        torch.save(self.net.state_dict(), './checkpoint.pkl')
                        checkpoint_counter = 0
                    if checkpoint_counter <= 50 and self.sticky == 0:
                        checkpoint_counter += 1
                            
                    net.load_state_dict(self.net.state_dict())
                    s1 = rewardwrapper.reset()
                    s1 = s1.reshape([1,s1.shape[0],s1.shape[1],3])
                    s1 = torch.from_numpy(s1).float()
                    s1 = s1.transpose(1,3)
                    s1 = F.interpolate(s1, scale_factor=0.25, mode='bilinear', align_corners=True)
                    poop = torch.zeros(1,n)
                    ep_score = 0
                    self.lock.release()
                    net.sample_noise()
                    net.zero_grad()
                    
                rewardwrapper.env.render()
                act = net.forward(s1, poop.round())
                pooper = torch.distributions.binomial.Binomial(logits=act)
                updater = torch.distributions.binomial.Binomial(probs=act.sigmoid().mul(2).add(-1).abs()*0.9+0.05)
                update = updater.sample()
                poop = pooper.sample().mul(update).add(poop.mul(1-update))
                    
                poopact = poop.round().squeeze().numpy().tolist()
                s1, score, done, info = rewardwrapper.step(poopact)
                ep_score += score
                s1 = s1.reshape([1,s1.shape[0],s1.shape[1],3])
                s1 = torch.from_numpy(s1).float()
                s1 = s1.transpose(1,3)
                s1 = F.interpolate(s1, scale_factor=0.25, mode='bilinear', align_corners=True)
                
torch.set_flush_denormal(True)

if __name__ == '__main__':
    n = 9
    lock = mp.Lock()
    net = acNet(n)
    net.share_memory()
    workers = [train_window(net,lock,sticky=i) for i in range(1)]
    [w.start() for w in workers]
    [w.join() for w in workers]
    
