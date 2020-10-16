from pathlib import Path
import argparse
import cv2
import matplotlib.cm as cm
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from models.utils import (AverageTimer, VideoStreamer,
                          make_matching_plot_fast, frame2tensor)
from tools import common
from tools.dataloader import norm_RGB
from nets.patchnet import *
'''
@function 加载网络
'''
def load_network(model_fn): 
    # 加载模型参数，返回的结果是一个dict字典类型
    checkpoint = torch.load(model_fn)
    print("\n>> Creating net = " + checkpoint['net']) 
    # 执行Quad_L2Net_ConfCFS()获得一个net类
    net = eval(checkpoint['net']) # checkpoint['net']) = "Quad_L2Net_ConfCFS()"

    nb_of_weights = common.model_size(net) #获得网络net的参数数量
    print(f" ( Model size: {nb_of_weights/1000:.0f}K parameters )")

    # initialization
    weights = checkpoint['state_dict']  #获取加载模型中的权重信息
    # load_state_dict函数是从一个有序字典数据结构来加载网络参数
    net.load_state_dict({k.replace('module.',''):v for k,v in weights.items()})
    return net.eval()   

'''
@class 非极大值抑制类，继承自torch.nn.Module；
@function 这个类比较简单，其实就只有一层max——pooling层，3*3核，步长为1
'''
class NonMaxSuppression (torch.nn.Module):
    def __init__(self, rel_thr=0.7, rep_thr=0.7):
        nn.Module.__init__(self)
        self.max_filter = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.rel_thr = rel_thr  #阈值
        self.rep_thr = rep_thr
    
    def forward(self, reliability, repeatability, **kw):
        assert len(reliability) == len(repeatability) == 1
        reliability, repeatability = reliability[0], repeatability[0]

        # local maxima，计算可重复性的局部最大（3*3区域）
        maxima = (repeatability == self.max_filter(repeatability))

        # remove low peaks 根据可靠性和可重复性是否大于阈值再筛去一些比较小的峰值
        maxima *= (repeatability >= self.rep_thr)   #可重复性与可靠性必须大于一定阈值
        maxima *= (reliability   >= self.rel_thr)

        return maxima.nonzero().t()[2:4]    # 返回不是0的位置

'''
@function 多尺度的提取特征
@params     net         网络
            img         输入的图像
            detector    非极大值抑制的网络
'''
def extract_singlescale( net, img, detector,
                        verbose=False):
    # 设置 torch.backends.cudnn.benchmark=True 将会让程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速。
    old_bm = torch.backends.cudnn.benchmark 
    torch.backends.cudnn.benchmark = False # speedup
    
    # extract keypoints at multiple scales
    # 验证是否是一个（1，3，H，W）的tensor，这样才是单张rgb图像，可以作为网络的输入
    B, three, H, W = img.shape
    assert B == 1 and three == 3, "should be a batch with a single RGB image"
    
    
    # 当前的尺寸必须大于最小的阈值
    with torch.no_grad():   # 为了节省资源，不会自动计算梯度
        res = net(imgs=[img])
                
    # get output and reliability map 
    descriptors = res['descriptors'][0] # 描述子集合（因为只有一张图，所以0）
    reliability = res['reliability'][0] # 可靠性map
    repeatability = res['repeatability'][0] #可重复性map

    # normalize the reliability for nms
    # extract maxima and descs
    y,x = detector(**res) # nms 执行非极大值抑制
    c = reliability[0,0,y,x]    # 可靠性的响应值集合
    q = repeatability[0,0,y,x]  # 可重复性的响应值集合
    d = descriptors[0,:,y,x].t()    # 描述子集合
    n = d.shape[0]  # 本轮检测到的特征的数量

    # restore value
    torch.backends.cudnn.benchmark = old_bm

    # 拼接成tensor后，对应值相乘得到对应点的综合分数
    scores = c * q #torch.cat(C) * torch.cat(Q) # scores = reliability * repeatability
    XY = torch.stack([x,y], dim=-1)  # stack函数是将对应维度的拼接起来放入一个tensor（-1表示最后一个维度），此处表达的意思是将每个特征的x,y,s拼起来成一个小tensor
    D = d #torch.cat(D)    
    return XY, D, scores   # 返回关键点的位置信息，描述子，评分        



torch.set_grad_enabled(False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='r2d2 demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--input', type=str, default='0',
        help='ID of a USB webcam, URL of an IP camera, '
             'or path to an image directory or movie file')
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Directory where to write output frames (If None, no output)')

    parser.add_argument(
        '--image_glob', type=str, nargs='+', default=['*.png', '*.jpg', '*.jpeg'],
        help='Glob if a directory of images is specified')
    parser.add_argument(
        '--skip', type=int, default=1,
        help='Images to skip if input is a movie or directory')
    parser.add_argument(
        '--max_length', type=int, default=1000000,
        help='Maximum length if input is a movie or directory')
    parser.add_argument(
        '--resize', type=int, nargs='+', default=[640, 480],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')

    parser.add_argument("--model", type=str, default='models/r2d2_WASF_N16.pt', help='model path')
    parser.add_argument("--top-k", type=int, default=5000, help='number of keypoints')

    parser.add_argument("--reliability-thr", type=float, default=0.7)
    parser.add_argument("--repeatability-thr", type=float, default=0.7)

    parser.add_argument(
        '--no_display', action='store_true',
        help='Do not display images to screen. Useful if running remotely')
    parser.add_argument(
        '--force_cpu', action='store_true',
        help='Force pytorch to run in CPU mode.')

    opt = parser.parse_args()
    print(opt)

    if len(opt.resize) == 2 and opt.resize[1] == -1:
        opt.resize = opt.resize[0:1]
    if len(opt.resize) == 2:
        print('Will resize to {}x{} (WxH)'.format(
            opt.resize[0], opt.resize[1]))
    elif len(opt.resize) == 1 and opt.resize[0] > 0:
        print('Will resize max dimension to {}'.format(opt.resize[0]))
    elif len(opt.resize) == 1:
        print('Will not resize images')
    else:
        raise ValueError('Cannot specify more than two integers for --resize')

    device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
    print('Running inference on device \"{}\"'.format(device))

    net = load_network(opt.model).to(device)
    detector = NonMaxSuppression(
        rel_thr = opt.reliability_thr, 
        rep_thr = opt.repeatability_thr)

    vs = VideoStreamer(opt.input, opt.resize, opt.skip,
                       opt.image_glob, opt.max_length)

    # Create a window to display the demo.
    if not opt.no_display:
        cv2.namedWindow('r2d2', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('r2d2', (640, 480))
    else:
        print('Skipping visualization, will not show a GUI.')

    # Print the keyboard help menu.
    print('==> Keyboard control:\n'
          '\tq: quit')

    #timer = AverageTimer()

    while True:
        frame, ret = vs.next_frame()
        w, h = frame.shape[1], frame.shape[0]
        id = vs.i-1
        if not ret:
            print('Finished demo_r2d2.py')
            break

        img = norm_RGB(frame)[None].to(device)
        
        #frame = frame.to(device)
        xys, desc, scores = extract_singlescale(net, img, detector,
            verbose = True)

        xys = xys.cpu().numpy() #变为numpy类型
        desc = desc.cpu().numpy()
        scores = scores.cpu().numpy()
        idxs = scores.argsort()[-opt.top_k or None:]

        xys = xys[idxs]
        desc = desc[idxs]
        scores = scores[idxs]

        out = 255*np.ones((h,w,3), np.uint8)
        out = frame
        #out = np.stack([out]*3, -1)

        kpts0 = np.round(xys).astype(int)

        white = (255,255,255)
        black = (0,0,0)
        for x,y in kpts0:
            cv2.circle(out, (x, y), 2, black, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x, y), 1, white, -1, lineType=cv2.LINE_AA)

        if not opt.no_display:
            cv2.imshow('r2d2', out)
            key = chr(cv2.waitKey(1) & 0xFF)
            if key == 'q':
                vs.cleanup()
                print('Exiting (via q) demo_point.py')
                break

        if opt.output_dir is not None:
            #stem = 'matches_{:06}_{:06}'.format(last_image_id, vs.i-1)
            stem = 'points_{:06}'.format(id)
            out_file = str(Path(opt.output_dir, stem + '.png'))
            print('\nWriting image to {}'.format(out_file))
            cv2.imwrite(out_file, out)

    cv2.destroyAllWindows()
    vs.cleanup()