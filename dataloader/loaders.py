from __future__ import print_function, division
import re
import os
import math
import torch
import random
import warnings
import numpy as np
from PIL import Image
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets, models

warnings.filterwarnings("ignore")

def extract_and_combine_numbers(name):

    numbers = re.findall(r'\d+', name)

    combined_numbers = ''.join(numbers)

    return combined_numbers



class AUGAN_scene1(Dataset):

    def __init__(self, maps=np.zeros(1), phase='train',
                 num1=0, num2=0,
                 data="data/",
                 numTx=80,
                 thresh=0.2,
                 transform=transforms.ToTensor()):

        if maps.size == 1:
            self.maps = np.arange(0, 700, 1, dtype=np.int16)
            np.random.seed(42)
            np.random.shuffle(self.maps)
        else:
            self.maps = maps

        print('当前输入场景：AUGAN_scene1')
        self.data = data
        self.numTx = numTx
        self.thresh = thresh
        self.transform = transform
        self.height = 256
        self.width = 256
        self.num1 = num1
        self.num2 = num2

        if phase == 'train':
            self.num1 = 0
            self.num2 = 500
        if phase == 'val':
            self.num1 = 501
            self.num2 = 600
        elif phase == 'test':
            self.num1 = 601
            self.num2 = 700

        self.simulation = self.data + "image/"
        self.build = self.data + "build/"
        self.antenna = self.data + "antenna/"

    def __len__(self):
        return (self.num2 - self.num1) * self.numTx

    def __getitem__(self, idx):

        idxr = np.floor(idx / self.numTx).astype(int)
        idxc = idx - idxr * self.numTx
        dataset_map = self.maps[idxr + self.num1]

        name1 = str(dataset_map) + ".png"

        name2 = str(dataset_map) + "_" + str(idxc) + ".png"

        # Loading building
        builds = os.path.join(self.build, name1)
        # Spin array
        arr_build = np.asarray(io.imread(builds))

        # Loading antenna
        antennas = os.path.join(self.antenna, name2)
        # Spin array
        arr_antenna = np.asarray(io.imread(antennas))

        # loading target
        target = os.path.join(self.simulation, name2)
        arr_target = np.asarray(io.imread(target))

        # threshold setting
        if self.thresh >= 0:
            arr_target = arr_target / 255
            mask = arr_target < self.thresh
            arr_target[mask] = self.thresh
            arr_target = arr_target - self.thresh * np.ones(np.shape(arr_target))
            arr_target = arr_target / (1 - self.thresh)

        # transfer tensor
        arr_builds = self.transform(arr_build).type(torch.float32)
        arr_antennas = self.transform(arr_antenna).type(torch.float32)
        arr_targets = self.transform(arr_target).type(torch.float32)

        return arr_builds, arr_antennas, arr_targets, name2

class AUGAN_scene2(Dataset):

    def __init__(self, maps=np.zeros(1), phase='train',
                 num1=0, num2=0,
                 data="data/",
                 numTx=80,
                 thresh=0.2,
                 sample=True,
                 uniform_fix_rate=True,
                 uniform_random_rate=True,
                 random_rate=True,
                 sample_rate=0.01,
                 transform=transforms.ToTensor()):

        if maps.size == 1:
            self.maps = np.arange(0, 700, 1, dtype=np.int16)
            np.random.seed(42)
            np.random.shuffle(self.maps)
        else:
            self.maps = maps

        self.data = data
        self.numTx = numTx
        self.thresh = thresh
        self.transform = transform
        self.height = 256
        self.width = 256
        self.sample = sample
        self.sample_rate = sample_rate
        self.uniform_fix_rate = uniform_fix_rate
        self.uniform_random_rate = uniform_random_rate
        self.random_rate = random_rate
        self.num1 = num1
        self.num2 = num2

        print('当前输入场景：AUGAN_scene2')


        if phase == 'train':
            self.num1 = 0
            self.num2 = 500
        if phase == 'val':
            self.num1 = 501
            self.num2 = 600
        elif phase == 'test':
            self.num1 = 600
            self.num2 = 700

        self.simulation = self.data + "image/"
        self.build = self.data + "build/"
        self.antenna = self.data + "antenna/"

    def __len__(self):
        return (self.num2 - self.num1) * self.numTx

    def __getitem__(self, idx):

        idxr = np.floor(idx / self.numTx).astype(int)
        idxc = idx - idxr * self.numTx
        dataset_map = self.maps[idxr + self.num1]

        name1 = str(dataset_map) + ".png"
        name2 = str(dataset_map) + "_" + str(idxc) + ".png"

        builds = os.path.join(self.build, name1)
        arr_build = np.asarray(io.imread(builds))

        antennas = os.path.join(self.antenna, name2)
        arr_antenna = np.asarray(io.imread(antennas))

        target = os.path.join(self.simulation, name2)
        arr_target = np.asarray(io.imread(target))

        if self.thresh >= 0:
            arr_target = arr_target / 255
            mask = arr_target < self.thresh
            arr_target[mask] = self.thresh
            arr_target = arr_target - self.thresh * np.ones(np.shape(arr_target))
            arr_target = arr_target / (1 - self.thresh)

        # sampling(True/False)
        if self.sample == True:
            # Fixed sampling rate
            if self.uniform_fix_rate == True:

                # print('1% 均匀采样')
                sample_num = int(np.round(math.sqrt(self.width * self.height * self.sample_rate)))
                arr_sample = np.zeros((self.width, self.height))
                step = int(np.round(self.width / sample_num))
                for _ in range(0, self.width, step):
                    for __ in range(0, self.height, step):
                        arr_sample[_][__] = (arr_target * 255)[_][__]

            # (Uniform)Random sampling rate(1%-10%)
            if self.uniform_random_rate == False:
                # print('1%-10%均匀采样')
                random_rate = random.uniform(0.01, 0.1)
                sample_num = int(np.round(math.sqrt(self.width * self.height * random_rate)))
                arr_sample = np.zeros((self.width, self.height))
                step = int(np.round(self.width / sample_num))
                for _ in range(0, self.width, step):
                    for __ in range(0, self.height, step):
                        arr_sample[_][__] = (arr_target * 255)[_][__]

            # (random) Random sampling rate(1%-10%)
            if self.random_rate == False:
                # print('1%-10%随机采样')
                arr_sample = np.zeros((self.width, self.height))
                random_rate = random.uniform(0.01, 0.1)
                sample_points = np.random.choice([0, 1], size=(self.width, self.height),
                                                 p=[1 - random_rate, random_rate])
                arr_sample[sample_points == 1] = 1
                arr_sample = arr_sample * (arr_target * 255)

        # transfer tensor
        arr_builds = self.transform(arr_build).type(torch.float32)
        arr_antennas = self.transform(arr_antenna).type(torch.float32)
        arr_targets = self.transform(arr_target).type(torch.float32)
        arr_sample = self.transform(arr_sample).type(torch.float32)

        return arr_builds, arr_antennas, arr_sample, arr_targets, name2

class AUGAN_scene3(Dataset):

    def __init__(self,maps=np.zeros(1), phase='train',
                 num1=0,num2=0,
                 data="data/",
                 numTx=80,
                 sample_size=4,
                 add_noise=False,
                 mean=0, sigma=10,                                  # Noise mean and standard deviation initialization
                 sample_num=100,
                 transform=transforms.ToTensor()):

        if maps.size == 1:
            self.maps = np.arange(0, 700, 1, dtype=np.int16)
            np.random.seed(42)
            np.random.shuffle(self.maps)
        else:
            self.maps = maps

        print('当前输入场景：AUGAN_scene3')
        self.data = data
        self.numTx = numTx
        self.transform = transform
        self.height = 256
        self.width = 256
        self.mean = mean
        self.sigma = sigma
        self.add_noise = add_noise
        self.sample_size = sample_size
        self.sample_num = sample_num
        self.num1 = num1
        self.num2 = num2

        if phase == 'train':
            self.num1 = 0
            self.num2 = 500
        elif phase == 'val':
            self.num1 = 501
            self.num2 = 600
        elif phase == 'test':
            self.num1 = 601
            self.num2 = 700

        self.simulation = self.data+"image/"
        self.build = self.data + "build/"
        self.antenna = self.data + "antenna/"

        
    def __len__(self):
        return (self.num2-self.num1)*self.numTx

    def __getitem__(self, idx):

        idxr = np.floor(idx/self.numTx).astype(int)
        idxc = idx-idxr*self.numTx
        dataset_map = self.maps[idxr+self.num1]

        name1 = str(dataset_map) + ".png"
        name2 = str(dataset_map) + "_" + str(idxc) + ".png"

        # loading target
        target_path = os.path.join(self.simulation, name2)
        target_image = Image.open(target_path)
        target_arr = np.asarray(target_image)

        # sampling
        numbers_combine = extract_and_combine_numbers(name2)

        x_seed = '1' + numbers_combine
        y_seed = '2' + numbers_combine

        # Build whiteboard diagram (to store sampling points)
        sample_image = Image.new("L", target_image.size, "black")

        num = 0
        # sample
        for i in range((self.width - self.sample_size) * (self.height - self.sample_size)):

            # Generate random points along the upper left corner
            random.seed(int(x_seed + str(i)))
            x = random.randint(0, self.width - self.sample_size)
            random.seed(int(y_seed + str(i)))
            y = random.randint(0, self.height - self.sample_size)

            # length * width
            block = target_image.crop((x, y, x + self.sample_size, y + self.sample_size))

            # Select the sample block area that meets the conditions
            if not np.any(np.any(np.array(block) == 0, axis=0)):
                if self.add_noise == True:
                    arr_block = np.asarray(block)
                    # noise
                    gaussian_noise = np.random.normal(self.mean, self.sigma, (4, 4))
                    # fuse
                    add_noise_block = arr_block + gaussian_noise
                    # transfer image
                    block = Image.fromarray(add_noise_block.astype(np.uint8))
                sample_image.paste(block, (x, y))
                num = num + 1
            # sample num
            if num == self.sample_num:
                break

        # Does not contain a sample of the building
        sample_arr = np.asarray(sample_image)

        # building image
        build_arr = np.where(target_arr == 0, 255, 0)

        # Contains a sampling of the building
        image_arr = sample_arr + build_arr

        # generate masks
        mask_arr = np.where(image_arr == 0, 255, 0)

        # transfer tensor
        arr_image = self.transform(image_arr / 255).type(torch.float32)
        arr_target = self.transform(target_arr).type(torch.float32)
        arr_mask = self.transform(mask_arr/255).type(torch.float32)

        return arr_image, arr_mask, arr_target, name2

# def test():
#     dataset = AOT_UNet1(phase='test')
#     loader = DataLoader(dataset, batch_size=5)
#
#     for x, y, z in loader:
#         print(x.shape, y.shape, z)
#
# if __name__ == "__main__":
#     test()






