import os
import torch
import logging
import cv2
from PIL import Image
import imageio
import numpy as np
import torch.utils.data as data
from os.path import join, exists
import math
import random
import sys
import json
import random
from subnet.basics import *
from subnet.ms_ssim_torch import ms_ssim
from augmentation import random_flip, random_crop_and_pad_image_and_labels
from torchvision.utils import save_image

class EWAP_ethDataSet(data.Dataset):
    def __init__(self, crf, gop, seq=['1']):
        folders = seq
        self.ref = []
        self.refbpp = []
        self.input = []
        self.hevcclass = []
        
        
        for seq in folders:
            seq = seq.rstrip()
            root = '/ai/base/data/wangfuchun/PyTorchVideoCompression/DVC/data/surveillance/ewap_eth'
            seqIbpp = 1.0
            imlist = os.listdir(os.path.join(root, seq))
            imlist.sort(key=lambda x: int(x.split('.')[0]))
            # print(imlist)
            # exit()
            
            cnt = 0
            for im in imlist:
                if im[-4:] == '.png':
                    cnt += 1

            sample_image_path = os.path.join(root, seq, imlist[0])
            import imagesize
            width, height = imagesize.get(sample_image_path)
            # print(f'{cnt=}')
            # exit()
            
            experiment_dir = f'/ai/base/data/wangfuchun/PyTorchVideoCompression/DVC/examples/pre_trained_dvc/data/surveillance/ffmpeg/h265/ewap_eth/{seq}/gop_experiments/crf_{crf}_gop_{gop}'
            print(f'{experiment_dir=}')
            log_file = os.path.join(experiment_dir, 'ffmpeg_compression.log')
            bpp = []
            # print(f'{log_file=}')
            # 输出视频文件大小
            # video_name = os.path.join(experiment_dir, 'compressed.mkv')
            # print(f'{video_name=}')
            # video_size = os.path.getsize(video_name)
            # print(f'{video_size=}')
            
            bpps = []
            for line in open(log_file, 'r'):
                if 'keyframe 1' not in line:
                    continue
                comma_index = line.find(', size')
                line = line[comma_index + 1:]
                comma_index = line.find(',')
                line = line[:comma_index]
                sz = int(line.split(' ')[-1])
                bpp.append(sz * 8 / (width * height))
            framerange = cnt // gop
            assert len(bpp) == framerange, 'len(bpp) != framerange'
            seqIbpp = np.mean(bpp)
            print(f'{seqIbpp=}')
            # exit()
            im_id = 0
            for i in range(framerange):
                refpath = f'{experiment_dir}/decompressed_image/{imlist[im_id]}'
                # print('refpath', refpath)
                # exit()
                self.ref.append(refpath)
                
                inputpath = []
                for j in range(gop):
                    inputpath.append(os.path.join(root, seq, imlist[im_id]))
                    im_id = im_id + 1
                
                self.refbpp.append(seqIbpp)
                self.input.append(inputpath)
            

    
    def __len__(self):
        return len(self.ref)

    def __getitem__(self, index):
        ref_image = imageio.imread(self.ref[index]).transpose(2, 0, 1).astype(np.float32) / 255.0
        h = (ref_image.shape[1] // 64) * 64
        w = (ref_image.shape[2] // 64) * 64
        ref_image = np.array(ref_image[:, :h, :w])

        input_images = []
        refpsnr = None
        refmsssim = None
        for filename in self.input[index]:
            input_image = (imageio.imread(filename).transpose(2, 0, 1)[:, :h, :w]).astype(np.float32) / 255.0
            if refpsnr is None:
                refpsnr = CalcuPSNR(input_image, ref_image)
                refmsssim = ms_ssim(torch.from_numpy(input_image[np.newaxis, :]), torch.from_numpy(ref_image[np.newaxis, :]), data_range=1.0).numpy()
            else:
                input_images.append(input_image[:, :h, :w])

        input_images = np.array(input_images)
        return input_images, ref_image, self.refbpp[index], refpsnr, refmsssim


# class EWAP_hotel_DataSet(data.Dataset):
#     def __init__(self, root="data/surveillance/ewap_hotel", filelist="data/UVG/originalv.txt", refdir='L12000', testfull=False, folder=['9']):
#         # with open(filelist) as f:
#         #     folders = f.readlines()
#         # folders = ['9']
#         folders = ['3']
#         # 3 号测试集 2 号验证集
#         # folders = ['0', '1', '2', '3','4', '5','6', '7', '8', '9']
#         self.ref = []
#         self.refbpp = []
#         self.input = []
#         self.hevcclass = []
#         # AllIbpp = self.getbpp(refdir)
#         ii = 0

#         from compressai.zoo import (bmshj2018_factorized, bmshj2018_hyperprior, mbt2018_mean, mbt2018, cheng2020_anchor)
#         from torchvision import transforms
#         device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         metric = 'mse'  # only pre-trained model for mse are available for now
#         quality = 6     # lower quality -> lower bit-rate (use lower quality to clearly see visual differences in the notebook)
#         # networks = {
#         #     'cheng2020-anchor': 
#         # }
#         net = cheng2020_anchor(quality=quality, pretrained=True).eval().to(device)

#         for folder in folders:
#             seq = folder.rstrip()
#             seqIbpp = 1.0

#             imlist = os.listdir(os.path.join(root, seq))
#             imlist = list(filter(lambda x: 'ref' not in x, imlist)) 
#             imlist = list(map(lambda x: int(x[:-4]), imlist))
#             imlist.sort()
#             imlist = list(map(lambda x: str(x)+'.png', imlist))
#             # print(imlist)
#             # exit()
#             cnt = 0
#             for im in imlist:
#                 if im[-4:] == '.png':
#                     cnt += 1
#             print(f'{cnt=}')
#             # exit()
            
#             gop = 10
#             framerange = cnt // gop
#             im_id = 0
#             for i in range(framerange):
#                 refpath = os.path.join(root, seq,  imlist[im_id])

#                 # compress_ref_image
#                 ref_image = imageio.imread(refpath).transpose(2, 0, 1).astype(np.float32) / 255.0
#                 h = (ref_image.shape[1] // 64) * 64
#                 w = (ref_image.shape[2] // 64) * 64
#                 ref_image = np.array(ref_image[:, :h, :w])
                
#                 x = torch.from_numpy(ref_image).to(device).unsqueeze(0)

#                 rv = net(x)
#                 rv['x_hat'].clamp_(0, 1)
#                 out = rv
#                 ref_image_hat = out['x_hat'].squeeze()

#                 refpath = refpath[:-4] + '_ref.png'
#                 save_image(ref_image_hat, refpath)

#                 psnr = compute_psnr(x, out["x_hat"])
#                 bpp = compute_bpp(out)
#                 seqIbpp = bpp
#                 # print(bpp, psnr)
#                 # exit()

#                 inputpath = []
#                 for j in range(gop):
#                     inputpath.append(os.path.join(root, seq, imlist[im_id]))
#                     im_id = im_id + 1
#                 self.ref.append(refpath)
#                 self.refbpp.append(seqIbpp)
#                 self.input.append(inputpath)

#             ii += 1

    
#     def __len__(self):
#         return len(self.ref)

#     def __getitem__(self, index):
#         ref_image = imageio.imread(self.ref[index]).transpose(2, 0, 1).astype(np.float32) / 255.0
#         h = (ref_image.shape[1] // 64) * 64
#         w = (ref_image.shape[2] // 64) * 64
#         ref_image = np.array(ref_image[:, :h, :w])


#         input_images = []
#         refpsnr = None
#         refmsssim = None
#         for filename in self.input[index]:
#             input_image = (imageio.imread(filename).transpose(2, 0, 1)[:, :h, :w]).astype(np.float32) / 255.0
#             if refpsnr is None:
#                 refpsnr = CalcuPSNR(input_image, ref_image)
#                 refmsssim = ms_ssim(torch.from_numpy(input_image[np.newaxis, :]), torch.from_numpy(ref_image[np.newaxis, :]), data_range=1.0).numpy()
#             else:
#                 input_images.append(input_image[:, :h, :w])

#         input_images = np.array(input_images)
#         return input_images, ref_image, self.refbpp[index], refpsnr, refmsssim




# class CUHK_DataSet(data.Dataset):
#     def __init__(self, root="data/surveillance/train_HK", filelist="data/UVG/originalv.txt", refdir='L12000', testfull=False, folder=['9']):
#         # with open(filelist) as f:
#         #     folders = f.readlines()
#         # folders = ['9']
#         folders = folder
#         # 3 号测试集 2 号验证集
#         self.ref = []
#         self.refbpp = []
#         self.input = []
#         self.hevcclass = []
#         # AllIbpp = self.getbpp(refdir)
#         ii = 0

#         from compressai.zoo import (bmshj2018_factorized, bmshj2018_hyperprior, mbt2018_mean, mbt2018, cheng2020_anchor)
#         from torchvision import transforms
#         device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         metric = 'mse'  # only pre-trained model for mse are available for now
#         quality = 6     # lower quality -> lower bit-rate (use lower quality to clearly see visual differences in the notebook)
#         # networks = {
#         #     'cheng2020-anchor': 
#         # }
#         net = cheng2020_anchor(quality=quality, pretrained=True).eval().to(device)

#         for folder in folders:
#             seq = folder.rstrip()
#             seqIbpp = 1.0
#             imlist = os.listdir(os.path.join(root, seq))
#             imlist = list(filter(lambda x: 'ref' not in x, imlist)) 
#             imlist = list(map(lambda x: int(x[:-4]), imlist))
#             imlist.sort()
#             imlist = list(map(lambda x: str(x)+'.png', imlist))

#             # print(imlist)
#             # exit()
#             cnt = 0
#             for im in imlist:
#                 if im[-4:] == '.png':
#                     cnt += 1
#             print(f'{cnt=}')
#             # exit()
            
#             gop = 10
#             framerange = cnt // gop
#             im_id = 0
#             for i in range(framerange):
#                 refpath = os.path.join(root, seq,  imlist[im_id])

#                 # compress_ref_image
#                 ref_image = imageio.imread(refpath).transpose(2, 0, 1).astype(np.float32) / 255.0
#                 h = (ref_image.shape[1] // 64) * 64
#                 w = (ref_image.shape[2] // 64) * 64
#                 ref_image = np.array(ref_image[:, :h, :w])
                
#                 x = torch.from_numpy(ref_image).to(device).unsqueeze(0)

#                 rv = net(x)
#                 rv['x_hat'].clamp_(0, 1)
#                 out = rv
#                 ref_image_hat = out['x_hat'].squeeze()

#                 refpath = refpath[:-4] + '_ref.png'
#                 save_image(ref_image_hat, refpath)

#                 psnr = compute_psnr(x, out["x_hat"])
             
#                 bpp = compute_bpp(out)
#                 seqIbpp = bpp
#                 # print(bpp, psnr)
#                 # exit()

#                 inputpath = []
#                 for j in range(gop):
#                     inputpath.append(os.path.join(root, seq, imlist[im_id]))
#                     im_id = im_id + 1
#                 self.ref.append(refpath)
#                 self.refbpp.append(seqIbpp)
#                 self.input.append(inputpath)

#             ii += 1

    
#     def __len__(self):
#         return len(self.ref)

#     def __getitem__(self, index):
#         ref_image = imageio.imread(self.ref[index]).transpose(2, 0, 1).astype(np.float32) / 255.0
#         h = (ref_image.shape[1] // 64) * 64
#         w = (ref_image.shape[2] // 64) * 64
#         ref_image = np.array(ref_image[:, :h, :w])


#         input_images = []
#         refpsnr = None
#         refmsssim = None
#         for filename in self.input[index]:
#             input_image = (imageio.imread(filename).transpose(2, 0, 1)[:, :h, :w]).astype(np.float32) / 255.0
#             if refpsnr is None:
#                 refpsnr = CalcuPSNR(input_image, ref_image)
#                 # print(ref_image.shape)
#                 # print(input_image.shape)
#                 # print(type(ref_image))
#                 # print(type(input_image))
#                 # ref_image = np.array(input_image)
#                 refmsssim = ms_ssim(torch.from_numpy(input_image[np.newaxis, :]), torch.from_numpy(ref_image[np.newaxis, :]), data_range=1.0).numpy()
#             else:
#                 input_images.append(input_image[:, :h, :w])

#         input_images = np.array(input_images)
#         return input_images, ref_image, self.refbpp[index], refpsnr, refmsssim
