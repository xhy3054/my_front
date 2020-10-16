# Copyright 2019-present NAVER Corp.
# CC BY-NC-SA 3.0
# Available only for non-commercial use

import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.sampler import *
from nets.repeatability_loss import *
from nets.reliability_loss import *

# 将多个loss组合起来
class MultiLoss (nn.Module):
    """ Combines several loss functions for convenience.
    *args: [loss weight (float), loss creator, ... ]
    
    Example: 初始化实例
        loss = MultiLoss( 1, MyFirstLoss(), 0.5, MySecondLoss() )
    """
    def __init__(self, *args, dbg=()):
        nn.Module.__init__(self)
        assert len(args) % 2 == 0, 'args must be a list of (float, loss)'
        # 将不同loss封装成列表形式
        self.weights = []
        self.losses = nn.ModuleList()
        for i in range(len(args)//2):
            weight = float(args[2*i+0])
            loss = args[2*i+1]
            assert isinstance(loss, nn.Module), "%s is not a loss!" % loss
            self.weights.append(weight)
            self.losses.append(loss)

    # loss的前向计算， **的参数（表示一个dict）自动将传入传输封装进一个dict
    def forward(self, select=None, **variables):
        assert not select or all(1<=n<=len(self.losses) for n in select)
        d = dict()
        cum_loss = 0
        # 依次进行每个loss的处理，loss_func是当前的loss，weight是当前loss所占的权重，三个loss是1：1：1的
        for num, (weight, loss_func) in enumerate(zip(self.weights, self.losses),1):
            if select is not None and num not in select: continue
            # 该loss的前向处理
            l = loss_func(**{k:v for k,v in variables.items()})
            if isinstance(l, tuple): #如果返回的是个tuple
                assert len(l) == 2 and isinstance(l[1], dict)
            else:
                l = l, {loss_func.name:l}   #将其变为一个tuple，两个元素，第一个是1元素tensor，第二个是一个1元素字典
            cum_loss = cum_loss + weight * l[0] #loss加权叠加
            for key,val in l[1].items():
                d['loss_'+key] = float(val)
        d['loss'] = float(cum_loss)
        return cum_loss, d






