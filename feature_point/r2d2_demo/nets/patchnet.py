# Copyright 2019-present NAVER Corp.
# CC BY-NC-SA 3.0
# Available only for non-commercial use

import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

# PatchNet的基础类
class BaseNet (nn.Module):
    """ Takes a list of images as input, and returns for each image:
        - a pixelwise descriptor 像素级别的描述
        - a pixelwise confidence 像素级别的提取
    """
    def softmax(self, ux):
        if ux.shape[1] == 1:    #如果ux的列数为1
            x = F.softplus(ux)  # 此处相当于逐个元素执行log(1+exp(x))
            return x / (1 + x)  # for sure in [0,1], much less plateaus than softmax
        elif ux.shape[1] == 2:  #如果ux的列数为2
            # 对n维输入张量运用Softmax函数，将张量的每个元素缩放到（0,1）区间且和为1，此处按行计算（dim=1）
            return F.softmax(ux, dim=1)[:,1:2] #返回结果的第二列

    # 标准化函数，分别对最终计算的描述子、可重复性层、可靠性层进行标准化操作
    def normalize(self, x, ureliability, urepeatability):
        return dict(descriptors = F.normalize(x, p=2, dim=1), # 得到最终的特征层，针对x每行执行L2_norm运算，x_i/sqrt(x_1^2+...x_n^2)
                    repeatability = self.softmax( urepeatability ), #得到最终的可重复性层
                    reliability = self.softmax( ureliability )) #得到最终的可靠性层

    def forward_one(self, x):
        raise NotImplementedError()

    def forward(self, imgs, **kw): #使用网络一次处理imgs中的每张图像
        res = [self.forward_one(img) for img in imgs]
        # merge all dictionaries into one 将所有的字典组合到一起（描述子放到一起，可靠性map放在一起，可重复性放在一起）
        res = {k:[r[k] for r in res if k in r] for k in {k for r in res for k in r}}
        return dict(res, imgs=imgs, **kw)   # 返回一个字典


# PatchNet网络类，下面三种为图像patch提取描述子的网络都是基于这个类，此类提供了一些网络的基础功能
class PatchNet (BaseNet):
    """ Helper class to construct a fully-convolutional network that
        extract a l2-normalized patch descriptor.
    """
    def __init__(self, inchan=3, dilated=True, dilation=1, bn=True, bn_affine=False):
        BaseNet.__init__(self)  #首先是初始化BaseNet
        #设置参数
        self.inchan = inchan    #输入通道数量
        self.curchan = inchan   #当前通道数量
        self.dilated = dilated
        self.dilation = dilation
        self.bn = bn
        self.bn_affine = bn_affine
        self.ops = nn.ModuleList([]) #网络存储的list

    def _make_bn(self, outd):   #返回一个归一化层，防止数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
        return nn.BatchNorm2d(outd, affine=self.bn_affine)

    '''
    @function 增加卷积层
    @params outd        输出通道数量
            k           卷积核大小
            stride      步长
            dilation    kernel间距（就是又是kernel并不是直接1*1的印在图像上，有可能2*2的kernel作用的是一个4*4的patch，此时这个参数为2）
            bn          卷积层后是否添加一个归一化层
            relu        是否添加relu层（非线性层）
    '''
    def _add_conv(self, outd, k=3, stride=1, dilation=1, bn=True, relu=True): #增加卷积层
        d = self.dilation * dilation
        if self.dilated: 
            conv_params = dict(padding=((k-1)*d)//2, dilation=d, stride=1) #构建一个dict作为后续构建网络的参数
            self.dilation *= stride
        else:
            conv_params = dict(padding=((k-1)*d)//2, dilation=d, stride=stride)
        self.ops.append( nn.Conv2d(self.curchan, outd, kernel_size=k, **conv_params) ) #添加卷积层
        if bn and self.bn: self.ops.append( self._make_bn(outd) )
        if relu: self.ops.append( nn.ReLU(inplace=True) )
        self.curchan = outd
    
    # 使用本网络执行一次前向处理
    def forward_one(self, x):
        assert self.ops, "You need to add convolutions first"
        for n,op in enumerate(self.ops):
            x = op(x)
        return self.normalize(x)

# L2_Net，通过32*32的图片patches计算出一个128维的描述子，17年的论文
class L2_Net (PatchNet):
    """ Compute a 128D descriptor for all overlapping 32x32 patches.
        From the L2Net paper (CVPR'17).
    """
    def __init__(self, dim=128, **kw ):
        PatchNet.__init__(self, **kw)
        add_conv = lambda n,**kw: self._add_conv((n*dim)//128,**kw)
        add_conv(32)
        add_conv(32)
        add_conv(64, stride=2)
        add_conv(64)
        add_conv(128, stride=2)
        add_conv(128)
        add_conv(128, k=7, stride=8, bn=False, relu=False)
        self.out_dim = dim

# Quad_L2Net类，这种网络基于L2_Net,但是将L2_net最后的8*8卷积层替换成了连续的3个2*2卷积层
class Quad_L2Net (PatchNet):
    """ Same than L2_Net, but replace the final 8x8 conv by 3 successive 2x2 convs.
    """
    def __init__(self, dim=128, mchan=4, relu22=False, **kw ):
        PatchNet.__init__(self, **kw)   #初始化基础的PatchNet
        # 构建网络结构，与L2_Net结构一致，只有最后一层变了
        self._add_conv(  8*mchan)  # 卷积核为3步长1输出通道32的卷积层     
        self._add_conv(  8*mchan)  # 卷积核为3步长1输出通道32的卷积层
        self._add_conv( 16*mchan, stride=2)   # 卷积核为3步长2输出通道64的卷积层
        self._add_conv( 16*mchan)  # 卷积核为3步长1输出通道64的卷积层
        self._add_conv( 32*mchan, stride=2)   # 卷积核为3步长2输出通道128的卷积层
        self._add_conv( 32*mchan)  # 卷积核为3步长1输出通道128的卷积层
        # replace last 8x8 convolution with 3 2x2 convolutions 后面的8*8卷积层使用3个2*2的卷积层替换
        self._add_conv( 32*mchan, k=2, stride=2, relu=relu22)    # 卷积核为2步长2输出通道128的卷积层
        self._add_conv( 32*mchan, k=2, stride=2, relu=relu22)    # 卷积核为2步长2输出通道128的卷积层
        self._add_conv(dim, k=2, stride=2, bn=False, relu=False) # 卷积核为2步长2输出通道128的卷积层（本层不归一化，也不relu），最后一层
        self.out_dim = dim


# Quad_L2Net_ConfCFS类，这种网络比Quad_L2Net多出2个执行层，分别表示可重复性和可靠性
class Quad_L2Net_ConfCFS (Quad_L2Net):
    """ Same than Quad_L2Net, with 2 confidence maps for repeatability and reliability.
    """
    def __init__(self, **kw ):
        Quad_L2Net.__init__(self, **kw) #首先新建一个Quad_L2Net
        # reliability classifier # 可靠性层的分类器（用于生成可靠性层）
        self.clf = nn.Conv2d(self.out_dim, 2, kernel_size=1)   # 卷积核为1输出通道2的卷积层（所以可靠性层使用softmax）
        # repeatability classifier: for some reasons it's a softplus, not a softmax!
        # Why? I guess it's a mistake that was left unnoticed in the code for a long time...
        # 可重复性层的分类器（用于生成可重复性层）
        self.sal = nn.Conv2d(self.out_dim, 1, kernel_size=1)   # 卷积核为1输出通道1的卷积层（可重复层使用softplus）

    # 前向
    def forward_one(self, x):
        assert self.ops, "You need to add convolutions first"
        for op in self.ops:
            x = op(x) #逐层作用输入
        # 最终输出的x的尺寸为（1，128，w, h）
        # compute the confidence maps
        ureliability = self.clf(x**2)   #计算出初始的可靠性层（不是最终的）
        urepeatability = self.sal(x**2) #计算出初始的可重复性层（不是最终的）
        return self.normalize(x, ureliability, urepeatability)  # 计算出最终的描述子tensor，可靠性层，可重复性层

'''
此处总体归纳一下r2d2网络结构：
输入：
单张RGB图像，一般是（1，3，H, W）维度的tensor 

第一层：


输出：
一个字典res, res['descriptors']为描述子集合，只有一张图像则使用[0]访问第一张图像的描述子层，以下类似；res['reliability']为可靠性map集合；res['repeatability']为可重复性map集合；

使用：通过对可重复层进行非极大值抑制（max pooling）获得初始
'''



