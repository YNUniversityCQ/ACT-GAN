import os
import torch
import importlib
import numpy as np
from PIL import Image
from utils.option import args
from dataloader import loaders
from torch.utils.data import Dataset, DataLoader

def main_worker(args):

    # 加载模型
    net = importlib.import_module('model.' + args.model)
    model = net.InpaintGenerator(args).cuda()
    model.load_state_dict(torch.load('experiments/G0245000.pt', map_location='cuda'))
    model.eval()

    # 加载测试数据
    test_data = loaders.AOT_UNet(phase='test')
    test_dataloader = DataLoader(test_data, shuffle=False, pin_memory=True, batch_size=1, num_workers=4)
    os.makedirs(args.outputs, exist_ok=True)

    interation = 0
    err1 = []
    err2 = []
    distance = []
    # 迭代测试
    for sample, mask, target, img_name in test_dataloader:
        interation += 1

        sample, mask, target = sample.cuda(), mask.cuda(), target.cuda()
        # sample = (target * (1 - mask).float()) + mask

        with torch.no_grad():
            predict_img = model(sample, mask)

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

        test3 = torch.tensor([item.cpu().detach().numpy() for item in mask]).cuda()
        test3 = test3.squeeze(0)
        test3 = test3.squeeze(0)
        mask = test3.cpu().numpy()
        image1 = np.where(image == 0, 255, image)
        sample = (1 - mask) * image1
        samples = Image.fromarray(sample.astype(np.uint8))

        # # 计算发射源位置的欧氏距离
        #
        # arr1 = np.asarray(predict)
        # arr2 = np.asarray(images)
        #
        # max_sum, max_area, size = -np.inf, None, 4
        #
        # # 数值和最大的 stride×stride 区域
        # for _ in range(254):
        #     for __ in range(254):
        #         kernel = arr1[_:_ + size, __:__ + size]
        #         kernel_sum = np.sum(kernel)
        #         if kernel_sum > max_sum:
        #             max_sum = kernel_sum
        #             max_area = kernel
        #             max_position = (_, __)
        #
        # # 找到最大数值和区域内的最大值
        # area_max = np.argmax(max_area)
        # # 索引最大数值和区域内最大值的坐标
        # max_index = np.unravel_index(area_max, max_area.shape)
        # # 最终的位置索引坐标
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

        # 计算均方根误差
        rmse = np.sqrt(np.mean((im2 - im1) ** 2))
        err1.append(rmse)
        # 计算平均均方误差
        nmse = np.mean((im2 - im1) ** 2)/np.mean((0 - im1) ** 2)
        err2.append(nmse)
        image_name = os.path.basename(img_name[0]).split('.')[0]
        images.save(os.path.join(args.outputs, f'{image_name}_target.png'))
        samples.save(os.path.join(args.outputs, f'{image_name}_sample.png'))
        predict.save(os.path.join(args.outputs, f'{image_name}_predict.png'))
        print(f'saving to {os.path.join(args.outputs, image_name)}', "RMSE:", rmse, "NMSE:", nmse)

        # 保存图片张数(总数为8000)
        if interation >= 8000:
            break
    rmse_err = sum(err1)/len(err1)
    nmse_err = sum(err2) / len(err2)
    # TX_distance = sum(distance) / len(distance)

    print('测试集均方根误差：', rmse_err)
    print('测试集归一化均方误差：', nmse_err)
    # print('测试集平均欧氏距离：', TX_distance)


if __name__ == '__main__':
    main_worker(args)
