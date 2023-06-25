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

class DataSet(data.Dataset):
    def __init__(self, path="data/vimeo_septuplet/test.txt", im_height=576, im_width=704):
        self.image_input_list, self.image_ref_list = self.get_vimeo(filefolderlist=path)
        self.im_height = im_height
        self.im_width = im_width
        
        self.featurenoise = torch.zeros([out_channel_M, self.im_height // 16, self.im_width // 16])
        self.znoise = torch.zeros([out_channel_N, self.im_height // 64, self.im_width // 64])
        self.mvnois = torch.zeros([out_channel_mv, self.im_height // 16, self.im_width // 16])
        print("dataset find image: ", len(self.image_input_list))

    def get_vimeo(self, rootdir="data/vimeo_septuplet/sequences/", filefolderlist="data/vimeo_septuplet/test.txt"):
 
        fns_train_input = []
        fns_train_ref = []


        root = '/ai/base/data/wangfuchun/PyTorchVideoCompression/DVC/examples/fg_bg_compression/data/surveillance'
        dataset = ['ewap_hotel']
        for d in dataset:
            path = os.path.join(root, d)
            folder_list = os.listdir(path)
            folder_list = list(filter(lambda x: x.isdigit(), folder_list)) 
            folder_list.sort()
            
            train_ratio = 0.7
            folder_list = folder_list[:int(len(folder_list)*0.7)]
            if d == 'ewap_hotel':
                folder_list = ['0', '1', '2', '4', '5','6', '7', '8', '9']

            print(d)
            print(folder_list)
            for folder in folder_list:
                folder_path = os.path.join(path, folder)
                print(f'{folder_path=}')
                imlist = os.listdir(folder_path)
                imlist = list(filter(lambda x: 'ref' not in x, imlist)) 
                imlist = list(map(lambda x: int(x[:-4]), imlist))
                imlist.sort()
                imlist = list(map(lambda x: str(x)+'.png', imlist))
                
                start_num = 1
                for i in range(start_num, len(imlist)):
                    im_name = imlist[i]
                    im_id = int(im_name[:-4])
                    y = os.path.join(folder_path, im_name)
                    fns_train_input += [y]
                    refnumber = im_id - start_num
                    refname = str(refnumber) + '.png'
                    fns_train_ref += [os.path.join(folder_path, refname)]

        return fns_train_input, fns_train_ref

    def __len__(self):
        return len(self.image_input_list)

    def __getitem__(self, index):
        input_image = imageio.imread(self.image_input_list[index])
        ref_image = imageio.imread(self.image_ref_list[index])

        input_image = input_image.astype(np.float32) / 255.0
        ref_image = ref_image.astype(np.float32) / 255.0

        input_image = input_image.transpose(2, 0, 1)
        ref_image = ref_image.transpose(2, 0, 1)
        
        input_image = torch.from_numpy(input_image).float()
        ref_image = torch.from_numpy(ref_image).float()


        input_image, ref_image = random_crop_and_pad_image_and_labels(input_image, ref_image, [self.im_height, self.im_width])

        # input_image, ref_image = random_flip(input_image, ref_image)
        quant_noise_feature, quant_noise_z, quant_noise_mv = torch.nn.init.uniform_(torch.zeros_like(self.featurenoise), -0.5, 0.5), torch.nn.init.uniform_(torch.zeros_like(self.znoise), -0.5, 0.5), torch.nn.init.uniform_(torch.zeros_like(self.mvnois), -0.5, 0.5)
        return input_image, ref_image, quant_noise_feature, quant_noise_z, quant_noise_mv
        




def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

def compute_bpp(out_net):
    size = out_net['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]
    return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
              for likelihoods in out_net['likelihoods'].values()).item()

class EWAP_ethDataSet(data.Dataset):
    def __init__(self, root="data/surveillance/ewap_eth", filelist="data/UVG/originalv.txt", refdir='L12000', testfull=False, folder=['1']):
        # with open(filelist) as f:
        #     folders = f.readlines()
        folders = folder
        # folders = ['2']
        # folders = ['0', '1', '2', '3','4', '5', '6']
        self.ref = []
        self.refbpp = []
        self.input = []
        self.hevcclass = []
        # AllIbpp = self.getbpp(refdir)
        ii = 0

        from compressai.zoo import (bmshj2018_factorized, bmshj2018_hyperprior, mbt2018_mean, mbt2018, cheng2020_anchor)
        from torchvision import transforms
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        metric = 'mse'  # only pre-trained model for mse are available for now
        quality = 6     # lower quality -> lower bit-rate (use lower quality to clearly see visual differences in the notebook)
        # networks = {
        #     'cheng2020-anchor': 
        # }
        net = cheng2020_anchor(quality=quality, pretrained=True).eval().to(device)

        for folder in folders:
            seq = folder.rstrip()
            seqIbpp = 1.0
            imlist = os.listdir(os.path.join(root, seq))
            imlist = list(filter(lambda x: 'ref' not in x, imlist)) 
            imlist = list(map(lambda x: int(x[:-4]), imlist))
            imlist.sort()
            imlist = list(map(lambda x: str(x)+'.png', imlist))
            # print(imlist)
            # exit()
            cnt = 0
            for im in imlist:
                if im[-4:] == '.png':
                    cnt += 1
            print(f'{cnt=}')
            # exit()
            
            # gop = 10
            gop = 25
            framerange = cnt // gop
            im_id = 0
            for i in range(framerange):
                refpath = os.path.join(root, seq,  imlist[im_id])

                # compress_ref_image
                # ref_image = imageio.imread(refpath).transpose(2, 0, 1).astype(np.float32) / 255.0
                # h = (ref_image.shape[1] // 64) * 64
                # w = (ref_image.shape[2] // 64) * 64
                # ref_image = np.array(ref_image[:, :h, :w])
                
                # x = torch.from_numpy(ref_image).to(device).unsqueeze(0)

                # rv = net(x)
                # rv['x_hat'].clamp_(0, 1)
                # out = rv
                # ref_image_hat = out['x_hat'].squeeze()

                # refpath = refpath[:-4] + '_ref.png'
                # save_image(ref_image_hat, refpath)

                # psnr = compute_psnr(x, out["x_hat"])
                # bpp = compute_bpp(out)
                # seqIbpp = bpp
                # print(bpp, psnr)
                # exit()

                crf = 19 
                refpath = f'/ai/base/data/wangfuchun/PyTorchVideoCompression/DVC/data/surveillance/ewap_eth/ffmpeg/out/h265/crf_{crf}/{folder}/' + imlist[im_id]
                image = imageio.imread(refpath)
                image_height, image_width = image.shape[0], image.shape[1]
                bpp = os.path.getsize(refpath) * 8 / (image_width * image_height)
                seqIbpp = 1.670505642361111
                # print('refpath', refpath)
                # exit()
                inputpath = []
                for j in range(gop):
                    inputpath.append(os.path.join(root, seq, imlist[im_id]))
                    im_id = im_id + 1
                
                self.ref.append(refpath)
                self.refbpp.append(seqIbpp)
                self.input.append(inputpath)
            ii += 1

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


class EWAP_hotel_DataSet(data.Dataset):
    def __init__(self, root="data/surveillance/ewap_hotel", filelist="data/UVG/originalv.txt", refdir='L12000', testfull=False, folder=['9']):
        # with open(filelist) as f:
        #     folders = f.readlines()
        # folders = ['9']
        folders = ['3']
        # 3 号测试集 2 号验证集
        # folders = ['0', '1', '2', '3','4', '5','6', '7', '8', '9']
        self.ref = []
        self.refbpp = []
        self.input = []
        self.hevcclass = []
        # AllIbpp = self.getbpp(refdir)
        ii = 0

        from compressai.zoo import (bmshj2018_factorized, bmshj2018_hyperprior, mbt2018_mean, mbt2018, cheng2020_anchor)
        from torchvision import transforms
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        metric = 'mse'  # only pre-trained model for mse are available for now
        quality = 6     # lower quality -> lower bit-rate (use lower quality to clearly see visual differences in the notebook)
        # networks = {
        #     'cheng2020-anchor': 
        # }
        net = cheng2020_anchor(quality=quality, pretrained=True).eval().to(device)

        for folder in folders:
            seq = folder.rstrip()
            seqIbpp = 1.0

            imlist = os.listdir(os.path.join(root, seq))
            imlist = list(filter(lambda x: 'ref' not in x, imlist)) 
            imlist = list(map(lambda x: int(x[:-4]), imlist))
            imlist.sort()
            imlist = list(map(lambda x: str(x)+'.png', imlist))
            # print(imlist)
            # exit()
            cnt = 0
            for im in imlist:
                if im[-4:] == '.png':
                    cnt += 1
            print(f'{cnt=}')
            # exit()
            
            gop = 10
            framerange = cnt // gop
            im_id = 0
            for i in range(framerange):
                refpath = os.path.join(root, seq,  imlist[im_id])

                # compress_ref_image
                ref_image = imageio.imread(refpath).transpose(2, 0, 1).astype(np.float32) / 255.0
                h = (ref_image.shape[1] // 64) * 64
                w = (ref_image.shape[2] // 64) * 64
                ref_image = np.array(ref_image[:, :h, :w])
                
                x = torch.from_numpy(ref_image).to(device).unsqueeze(0)

                rv = net(x)
                rv['x_hat'].clamp_(0, 1)
                out = rv
                ref_image_hat = out['x_hat'].squeeze()

                refpath = refpath[:-4] + '_ref.png'
                save_image(ref_image_hat, refpath)

                psnr = compute_psnr(x, out["x_hat"])
                bpp = compute_bpp(out)
                seqIbpp = bpp
                # print(bpp, psnr)
                # exit()

                inputpath = []
                for j in range(gop):
                    inputpath.append(os.path.join(root, seq, imlist[im_id]))
                    im_id = im_id + 1
                self.ref.append(refpath)
                self.refbpp.append(seqIbpp)
                self.input.append(inputpath)

            ii += 1

    
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




class CUHK_DataSet(data.Dataset):
    def __init__(self, root="data/surveillance/train_HK", filelist="data/UVG/originalv.txt", refdir='L12000', testfull=False, folder=['9']):
        # with open(filelist) as f:
        #     folders = f.readlines()
        # folders = ['9']
        folders = folder
        # 3 号测试集 2 号验证集
        self.ref = []
        self.refbpp = []
        self.input = []
        self.hevcclass = []
        # AllIbpp = self.getbpp(refdir)
        ii = 0

        from compressai.zoo import (bmshj2018_factorized, bmshj2018_hyperprior, mbt2018_mean, mbt2018, cheng2020_anchor)
        from torchvision import transforms
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        metric = 'mse'  # only pre-trained model for mse are available for now
        quality = 6     # lower quality -> lower bit-rate (use lower quality to clearly see visual differences in the notebook)
        # networks = {
        #     'cheng2020-anchor': 
        # }
        net = cheng2020_anchor(quality=quality, pretrained=True).eval().to(device)

        for folder in folders:
            seq = folder.rstrip()
            seqIbpp = 1.0
            imlist = os.listdir(os.path.join(root, seq))
            imlist = list(filter(lambda x: 'ref' not in x, imlist)) 
            imlist = list(map(lambda x: int(x[:-4]), imlist))
            imlist.sort()
            imlist = list(map(lambda x: str(x)+'.png', imlist))

            # print(imlist)
            # exit()
            cnt = 0
            for im in imlist:
                if im[-4:] == '.png':
                    cnt += 1
            print(f'{cnt=}')
            # exit()
            
            gop = 10
            framerange = cnt // gop
            im_id = 0
            for i in range(framerange):
                refpath = os.path.join(root, seq,  imlist[im_id])

                # compress_ref_image
                ref_image = imageio.imread(refpath).transpose(2, 0, 1).astype(np.float32) / 255.0
                h = (ref_image.shape[1] // 64) * 64
                w = (ref_image.shape[2] // 64) * 64
                ref_image = np.array(ref_image[:, :h, :w])
                
                x = torch.from_numpy(ref_image).to(device).unsqueeze(0)

                rv = net(x)
                rv['x_hat'].clamp_(0, 1)
                out = rv
                ref_image_hat = out['x_hat'].squeeze()

                refpath = refpath[:-4] + '_ref.png'
                save_image(ref_image_hat, refpath)

                psnr = compute_psnr(x, out["x_hat"])
             
                bpp = compute_bpp(out)
                seqIbpp = bpp
                # print(bpp, psnr)
                # exit()

                inputpath = []
                for j in range(gop):
                    inputpath.append(os.path.join(root, seq, imlist[im_id]))
                    im_id = im_id + 1
                self.ref.append(refpath)
                self.refbpp.append(seqIbpp)
                self.input.append(inputpath)

            ii += 1

    
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
                # print(ref_image.shape)
                # print(input_image.shape)
                # print(type(ref_image))
                # print(type(input_image))
                # ref_image = np.array(input_image)
                refmsssim = ms_ssim(torch.from_numpy(input_image[np.newaxis, :]), torch.from_numpy(ref_image[np.newaxis, :]), data_range=1.0).numpy()
            else:
                input_images.append(input_image[:, :h, :w])

        input_images = np.array(input_images)
        return input_images, ref_image, self.refbpp[index], refpsnr, refmsssim
