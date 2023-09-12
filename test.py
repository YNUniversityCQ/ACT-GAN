import os
import torch
import importlib
import numpy as np
from PIL import Image
from utils.option import args
from dataloader import loaders
from torch.utils.data import Dataset, DataLoader

def main_worker(args):

    # loading model
    net = importlib.import_module('model.' + args.model)
    model = net.Generator(args).cuda()
    model.load_state_dict(torch.load('experiments/G0405000.pt', map_location='cuda'))
    model.eval()

    # loading test data
    test_data = loaders.AUGAN_scene1(phase='test')  # AUGAN_scene1, AUGAN_scene2, AUGAN_scene3
    test_dataloader = DataLoader(test_data, shuffle=False, pin_memory=True, batch_size=1, num_workers=4)
    os.makedirs(args.outputs, exist_ok=True)

    interation = 0
    err1 = []
    err2 = []
    distance = []

    # scene 1
    for build, antenna, target, img_name in test_dataloader:
    # scene 2
    # for build, antenna, sample, target, img_name in test_dataloader:
    # scene 3
    # for sample, mask, target, img_name in test_dataloader:

        interation += 1

        # scene 1
        builds, antennas, targets = build.cuda(), antenna.cuda(), target.cuda()
        # scene 2
        # builds, antennas, samples, targets = build.cuda(), antenna.cuda(), sample.cuda(), target.cuda()
        # scene 3
        # samples, masks, targets = sample.cuda(), mask.cuda(), target.cuda()

        with torch.no_grad():

            # scene 1
            predict_img = model(builds, antennas)
            # scene 2
            # predict_img = model(builds, antennas, samples)
            # scene 3
            # predict_img = model(samples, masks)

        test = torch.tensor([item.cpu().detach().numpy() for item in predict_img]).cuda()
        test = test.squeeze(0)
        test = test.squeeze(0)
        im2 = test.cpu().numpy()
        im = test.cpu().numpy()*255
        predict = Image.fromarray(im.astype(np.uint8))

        test1 = torch.tensor([item.cpu().detach().numpy() for item in target]).cuda()
        test1 = test1.squeeze(0)
        test1 = test1.squeeze(0)
        im1 = test1.cpu().numpy()
        image = test1.cpu().numpy()*255
        images = Image.fromarray(image.astype(np.uint8))

        # # Location of the transmitting source

        # arr1 = np.asarray(predict)
        # arr2 = np.asarray(images)
        #
        # max_sum, max_area, size = -np.inf, None, 4
        #
        # # Value and maximum stride×stride area
        # for _ in range(254):
        #     for __ in range(254):
        #         kernel = arr1[_:_ + size, __:__ + size]
        #         kernel_sum = np.sum(kernel)
        #         if kernel_sum > max_sum:
        #             max_sum = kernel_sum
        #             max_area = kernel
        #             max_position = (_, __)
        #
        # area_max = np.argmax(max_area)
        # max_index = np.unravel_index(area_max, max_area.shape)
        # tx1_index = [max_position[0] + max_index[0], max_position[1] + max_index[1]]
        #
        # # print("原图像最大区域左上角位置:", max_position)
        # # print("Max Region:")
        # # print(max_area)
        # # print("Max Value:", max_index)
        #
        # tx2 = np.argmax(arr2)
        # tx2_index = np.unravel_index(tx2, arr2.shape)
        #
        # distances = np.linalg.norm(np.asarray(tx1_index) - np.asarray(tx2_index))
        #
        # distance.append(distances)

        # Calculate root-mean-square error
        rmse = np.sqrt(np.mean((im2 - im1) ** 2))
        err1.append(rmse)

        # Calculate the normalized mean square error
        nmse = np.mean((im2 - im1) ** 2)/np.mean((0 - im1) ** 2)
        err2.append(nmse)

        image_name = os.path.basename(img_name[0]).split('.')[0]
        images.save(os.path.join(args.outputs, f'{image_name}_target.png'))
        predict.save(os.path.join(args.outputs, f'{image_name}_predict.png'))
        print(f'saving to {os.path.join(args.outputs, image_name)}', "RMSE:", rmse, "NMSE:", nmse)

        # Number of pictures saved (total 8000)
        if interation >= 8000:
            break
    rmse_err = sum(err1)/len(err1)
    nmse_err = sum(err2) / len(err2)
    TX_distance = sum(distance) / len(distance)

    print('test RMSE：', rmse_err)
    print('test NMSE：', nmse_err)
    # print('test set mean Euclidean distance：', TX_distance)


if __name__ == '__main__':
    main_worker(args)
