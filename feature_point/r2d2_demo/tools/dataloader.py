# Copyright 2019-present NAVER Corp.
# CC BY-NC-SA 3.0
# Available only for non-commercial use

import pdb
from PIL import Image
import numpy as np

import torch
import torchvision.transforms as tvf

from tools.transforms import instanciate_transformation
from tools.transforms_tools import persp_apply


RGB_mean = [0.485, 0.456, 0.406]
RGB_std  = [0.229, 0.224, 0.225]

# compose将这些图像预处理操作连起来。
# ToTensor() ，将一个PIL图像转换为tensor
# Normalize函数用均值mean和标准差std对图像做归一化处理
norm_RGB = tvf.Compose([tvf.ToTensor(), tvf.Normalize(mean=RGB_mean, std=RGB_std)])


class PairLoader:
    """ On-the-fly jittering of pairs of image with dense pixel ground-truth correspondences.
    
    crop:   random crop applied to both images 应用于两张图
    scale:  random scaling applied to img2  只应用于第二张图
    distort: random ditorsion applied to img2   只应用于第二张图
    
    可以根据self[idx]的模式返回一个字典
    self[idx] returns a dictionary with keys: img1, img2, aflow, mask
     - img1: cropped original
     - img2: distorted cropped original
     - aflow: 'absolute' optical flow = (x,y) position of each pixel from img1 in img2
     - mask: (binary image) valid pixels of img1 二进制图像，每个像素显示图1中该位置像素是否有效
    """
    def __init__(self, dataset, crop='', scale='', distort='', norm = norm_RGB, 
                       what = 'aflow mask', idx_as_rng_seed = False):
        assert hasattr(dataset, 'npairs')
        assert hasattr(dataset, 'get_pair')
        self.dataset = dataset
        self.distort = instanciate_transformation(distort)
        self.crop = instanciate_transformation(crop)
        self.norm = instanciate_transformation(norm)
        self.scale = instanciate_transformation(scale)
        self.idx_as_rng_seed = idx_as_rng_seed # to remove randomness
        self.what = what.split() if isinstance(what, str) else what
        self.n_samples = 5 # number of random trials per image 每幅图的随机试验次数

    #返回数据集长度
    def __len__(self):
        assert len(self.dataset) == self.dataset.npairs, pdb.set_trace() # and not nimg
        return len(self.dataset)

    def __repr__(self):
        fmt_str = 'PairLoader\n'
        fmt_str += repr(self.dataset)
        fmt_str += '  npairs: %d\n' % self.dataset.npairs
        short_repr = lambda s: repr(s).strip().replace('\n',', ')[14:-1].replace('    ',' ')
        fmt_str += '  Distort: %s\n' % short_repr(self.distort)
        fmt_str += '  Crop: %s\n' % short_repr(self.crop)
        fmt_str += '  Norm: %s\n' % short_repr(self.norm)
        return fmt_str

    #支持索引
    def __getitem__(self, i):
        #from time import time as now; t0 = now()
        if self.idx_as_rng_seed:
            import random
            random.seed(i)
            np.random.seed(i)

        # Retrieve an image pair and their absolute flow
        # 检索出图像对和绝对flow
        img_a, img_b, metadata = self.dataset.get_pair(i, self.what)
        
        # aflow contains pixel coordinates indicating where each 
        # pixel from the left image ended up in the right image
        # as (x,y) pairs, but its shape is (H,W,2)
        # 获得绝对flow和mask
        aflow = np.float32(metadata['aflow'])
        mask = metadata.get('mask', np.ones(aflow.shape[:2],np.uint8))

        # apply transformations to the second image
        # 对第二副图进行变换
        img_b = {'img': img_b, 'persp':(1,0,0,0,1,0,0,0)}
        if self.scale:
            img_b = self.scale(img_b)
        if self.distort:
            img_b = self.distort(img_b)
        
        # apply the same transformation to the flow
        # 对光流进行相同的变换
        aflow[:] = persp_apply(img_b['persp'], aflow.reshape(-1,2)).reshape(aflow.shape)
        corres = None
        if 'corres' in metadata:
            corres = np.float32(metadata['corres'])
            corres[:,1] = persp_apply(img_b['persp'], corres[:,1])
        
        # apply the same transformation to the homography
        #对单应进行相同的变换
        homography = None
        if 'homography' in metadata:
            homography = np.float32(metadata['homography'])
            # p_b = homography * p_a
            persp = np.float32(img_b['persp']+(1,)).reshape(3,3)
            homography = persp @ homography

        # determine crop size
        img_b = img_b['img']
        crop_size = self.crop({'imsize':(10000,10000)})['imsize']
        output_size_a = min(img_a.size, crop_size)
        output_size_b = min(img_b.size, crop_size)
        img_a = np.array(img_a)
        img_b = np.array(img_b)

        ah,aw,p1 = img_a.shape
        bh,bw,p2 = img_b.shape
        assert p1 == 3
        assert p2 == 3
        assert aflow.shape == (ah, aw, 2)
        assert mask.shape == (ah, aw)

        # Let's start by computing the scale of the
        # optical flow and applying a median filter:
        dx = np.gradient(aflow[:,:,0])
        dy = np.gradient(aflow[:,:,1])
        scale = np.sqrt(np.clip(np.abs(dx[1]*dy[0] - dx[0]*dy[1]), 1e-16, 1e16))

        accu2 = np.zeros((16,16), bool)
        Q = lambda x, w: np.int32(16 * (x - w.start) / (w.stop - w.start))
        
        def window1(x, size, w):
            l = x - int(0.5 + size / 2)
            r = l + int(0.5 + size)
            if l < 0: l,r = (0, r - l)
            if r > w: l,r = (l + w - r, w)
            if l < 0: l,r = 0,w # larger than width
            return slice(l,r)
        def window(cx, cy, win_size, scale, img_shape):
            return (window1(cy, win_size[1]*scale, img_shape[0]), 
                    window1(cx, win_size[0]*scale, img_shape[1]))

        n_valid_pixel = mask.sum()
        sample_w = mask / (1e-16 + n_valid_pixel)
        def sample_valid_pixel():
            n = np.random.choice(sample_w.size, p=sample_w.ravel())
            y, x = np.unravel_index(n, sample_w.shape)
            return x, y
        
        # Find suitable left and right windows
        trials = 0 # take the best out of few trials
        best = -np.inf, None
        for _ in range(50*self.n_samples):
            if trials >= self.n_samples: break # finished!

            # pick a random valid point from the first image
            if n_valid_pixel == 0: break
            c1x, c1y = sample_valid_pixel()
            
            # Find in which position the center of the left
            # window ended up being placed in the right image
            c2x, c2y = (aflow[c1y, c1x] + 0.5).astype(np.int32)
            if not(0 <= c2x < bw and 0 <= c2y < bh): continue

            # Get the flow scale
            sigma = scale[c1y, c1x]

            # Determine sampling windows
            if 0.2 < sigma < 1: 
                win1 = window(c1x, c1y, output_size_a, 1/sigma, img_a.shape)
                win2 = window(c2x, c2y, output_size_b, 1, img_b.shape)
            elif 1 <= sigma < 5:
                win1 = window(c1x, c1y, output_size_a, 1, img_a.shape)
                win2 = window(c2x, c2y, output_size_b, sigma, img_b.shape)
            else:
                continue # bad scale

            # compute a score based on the flow
            x2,y2 = aflow[win1].reshape(-1, 2).T.astype(np.int32)
            # Check the proportion of valid flow vectors
            valid = (win2[1].start <= x2) & (x2 < win2[1].stop) \
                  & (win2[0].start <= y2) & (y2 < win2[0].stop)
            score1 = (valid * mask[win1].ravel()).mean()
            # check the coverage of the second window
            accu2[:] = False
            accu2[Q(y2[valid],win2[0]), Q(x2[valid],win2[1])] = True
            score2 = accu2.mean()
            # Check how many hits we got
            score = min(score1, score2)

            trials += 1
            if score > best[0]:
                best = score, win1, win2
        
        if None in best: # counldn't find a good window
            img_a = np.zeros(output_size_a[::-1]+(3,), dtype=np.uint8)
            img_b = np.zeros(output_size_b[::-1]+(3,), dtype=np.uint8)
            aflow = np.nan * np.ones((2,)+output_size_a[::-1], dtype=np.float32)
            homography = np.nan * np.ones((3,3), dtype=np.float32)

        else:
            win1, win2 = best[1:]
            img_a = img_a[win1]
            img_b = img_b[win2]
            aflow = aflow[win1] - np.float32([[[win2[1].start, win2[0].start]]])
            mask = mask[win1]
            aflow[~mask.view(bool)] = np.nan # mask bad pixels!
            aflow = aflow.transpose(2,0,1) # --> (2,H,W)
            
            if corres is not None:
                corres[:,0] -= (win1[1].start, win1[0].start)
                corres[:,1] -= (win2[1].start, win2[0].start)
            
            if homography is not None:
                trans1 = np.eye(3, dtype=np.float32)
                trans1[:2,2] = (win1[1].start, win1[0].start)
                trans2 = np.eye(3, dtype=np.float32)
                trans2[:2,2] = (-win2[1].start, -win2[0].start)
                homography = trans2 @ homography @ trans1
                homography /= homography[2,2]
            
            # rescale if necessary
            if img_a.shape[:2][::-1] != output_size_a:
                sx, sy = (np.float32(output_size_a)-1)/(np.float32(img_a.shape[:2][::-1])-1)
                img_a = np.asarray(Image.fromarray(img_a).resize(output_size_a, Image.ANTIALIAS))
                mask = np.asarray(Image.fromarray(mask).resize(output_size_a, Image.NEAREST))
                afx = Image.fromarray(aflow[0]).resize(output_size_a, Image.NEAREST)
                afy = Image.fromarray(aflow[1]).resize(output_size_a, Image.NEAREST)
                aflow = np.stack((np.float32(afx), np.float32(afy)))
                
                if corres is not None:
                    corres[:,0] *= (sx, sy)
                
                if homography is not None:
                    homography = homography @ np.diag(np.float32([1/sx,1/sy,1]))
                    homography /= homography[2,2]

            if img_b.shape[:2][::-1] != output_size_b:
                sx, sy = (np.float32(output_size_b)-1)/(np.float32(img_b.shape[:2][::-1])-1)
                img_b = np.asarray(Image.fromarray(img_b).resize(output_size_b, Image.ANTIALIAS))
                aflow *= [[[sx]], [[sy]]]
                
                if corres is not None:
                    corres[:,1] *= (sx, sy)
                
                if homography is not None:
                    homography = np.diag(np.float32([sx,sy,1])) @ homography
                    homography /= homography[2,2]
    
        assert aflow.dtype == np.float32, pdb.set_trace()
        assert homography is None or homography.dtype == np.float32, pdb.set_trace()
        if 'flow' in self.what:
            H, W = img_a.shape[:2]
            mgrid = np.mgrid[0:H, 0:W][::-1].astype(np.float32)
            flow = aflow - mgrid
        
        result = dict(img1=self.norm(img_a), img2=self.norm(img_b))
        for what in self.what:
            try: result[what] = eval(what)
            except NameError: pass
        return result



def threaded_loader( loader, iscuda, threads, batch_size=1, shuffle=True):
    """ Get a data loader, given the dataset and some parameters.
    
    Parameters
    ----------
    loader : object[i] returns the i-th training example.   其实torch类似dataset的一种类对象
    
    iscuda : bool   是否使用cuda
        
    batch_size : int    批量尺寸
    
    threads : int       线程数
    
    shuffle : bool      返回的patch中是否重新打乱数据
    
    Returns
    -------
        a multi-threaded pytorch loader.
    """
    '''
    `class torch.utils.data.DataLoader( dataset, 
                                        batch_size=1, 
                                        shuffle=False, 
                                        sampler=None, 
                                        num_workers=0, 
                                        collate_fn=<function default_collate>, 
                                        pin_memory=False, 
                                        drop_last=False)`
        dataset (Dataset) – 加载数据的数据集。
        batch_size (int, optional) – 每个batch加载多少个样本(默认: 1)。
        shuffle (bool, optional) – 设置为True时会在每个epoch重新打乱数据(默认: False).
        sampler (Sampler, optional) – 定义从数据集中提取样本的策略。如果指定，则忽略shuffle参数。
        num_workers (int, optional) – 用多少个子进程加载数据。0表示数据将在主进程中加载(默认: 0)
        collate_fn (callable, optional) –
        pin_memory (bool, optional) –
        drop_last (bool, optional) – 如果数据集大小不能被batch size整除，则设置为True后可删除最后一个不完整的batch。如果设为False并且数据集的大小不能被batch size整除，则最后一个batch将更小。(默认: False)    
    '''
    return torch.utils.data.DataLoader(
        loader,
        batch_size = batch_size,
        shuffle = shuffle,
        sampler = None,
        num_workers = threads,
        pin_memory = iscuda,
        collate_fn=collate)



def collate(batch, _use_shared_memory=True):
    """Puts each data field into a tensor with outer dimension batch size.
    Copied from https://github.com/pytorch in torch/utils/data/_utils/collate.py
    """
    import re
    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        out = None
        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        assert elem_type.__name__ == 'ndarray'
        # array of string classes and object
        if re.search('[SaUO]', elem.dtype.str) is not None:
            raise TypeError(error_msg.format(elem.dtype))
        batch = [torch.from_numpy(b) for b in batch]
        try:
            return torch.stack(batch, 0)
        except RuntimeError:
            return batch
    elif batch[0] is None:
        return list(batch)
    elif isinstance(batch[0], int):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], str):
        return batch
    elif isinstance(batch[0], dict):
        return {key: collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))



def tensor2img(tensor, model=None):
    """ convert back a torch/numpy tensor to a PIL Image
        by undoing the ToTensor() and Normalize() transforms.
    """
    mean = norm_RGB.transforms[1].mean
    std =  norm_RGB.transforms[1].std
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    
    res = np.uint8(np.clip(255*((tensor.transpose(1,2,0) * std) + mean), 0, 255))
    from PIL import Image
    return Image.fromarray(res)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser("Tool to debug/visualize the data loader")
    parser.add_argument("dataloader", type=str, help="command to create the data loader")
    args = parser.parse_args()

    from datasets import *
    auto_pairs = lambda db: SyntheticPairDataset(db,
        'RandomScale(256,1024,can_upscale=True)', 
        'RandomTilting(0.5), PixelNoise(25)')
        
    loader = eval(args.dataloader)
    print("Data loader =", loader)

    from tools.viz import show_flow
    for data in loader:
        aflow = data['aflow']
        H, W = aflow.shape[-2:]
        flow = (aflow - np.mgrid[:H, :W][::-1]).transpose(1,2,0)
        show_flow(tensor2img(data['img1']), tensor2img(data['img2']), flow)

