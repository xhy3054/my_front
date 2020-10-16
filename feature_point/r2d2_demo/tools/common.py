# Copyright 2019-present NAVER Corp.
# CC BY-NC-SA 3.0
# Available only for non-commercial use

import os, pdb#, shutil
import numpy as np
import torch


def mkdir_for(file_path):
    os.makedirs(os.path.split(file_path)[0], exist_ok=True)

# 计算输入的模型参数的数量
def model_size(model):
    ''' Computes the number of parameters of the model 
    '''
    size = 0
    # 遍历model的state_dict中的每个键值对中的值，模型调用state_dict()函数会返回一个有序字典，字典由网络的各个层组成，key是层的名字，value是层的参数集合
    for weights in model.state_dict().values():
        size += np.prod(weights.shape) # np.prod是对输入累乘所有元素，输入的是每个层的维度信息，得到该层参数数量
    return size

# 设置在什么机器上跑
def torch_set_gpu(gpus):
    if type(gpus) is int:
        gpus = [gpus] #放到一个序列中

    cuda = all(gpu>=0 for gpu in gpus)  #取出序列中所有大于等于0的gpu编号，得到一个gpu列表

    if cuda:    #如果gpu列表不为空，此处仅仅是判断一下cuda是否可用，并打印一下列表中的gpu编号，声明会在这些gpu上进行训练
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(gpu) for gpu in gpus])
        assert cuda and torch.cuda.is_available(), "%s has GPUs %s unavailable" % (
            os.environ['HOSTNAME'],os.environ['CUDA_VISIBLE_DEVICES'])
        torch.backends.cudnn.benchmark = True # speed-up cudnn
        torch.backends.cudnn.fastest = True # even more speed-up?
        print( 'Launching on GPUs ' + os.environ['CUDA_VISIBLE_DEVICES'] )

    else:
        print( 'Launching on CPU' )

    return cuda

