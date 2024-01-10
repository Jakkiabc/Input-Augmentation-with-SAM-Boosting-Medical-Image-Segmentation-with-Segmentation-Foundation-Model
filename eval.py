import numpy as np
import torch
import cv2
from model import UNet
from torchvision import transforms
from PIL import Image
import os

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((1024, 1024)),
    transforms.Normalize((0.5,), (0.5))
])

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = UNet(in_channels=3, num_classes=1)
net.load_state_dict(torch.load('/root/Unet/Unet/pth/one-UNet_DRIVE_ORI_5.pth', map_location=device))
net.to(device)

# 测试模式
net.eval()

root = r'/root/Unet/Unet/DRIVE/test/images'
#root = r'E:\Unet\CHASEDB1\test\images'
imgs_names = os.listdir(root)  # 读取图像的路径
imgs_path = [os.path.join(root, name) for name in imgs_names]
imgs_path=sorted(imgs_path)

# sam_mask_root = r'E:\Unet\DRIVE\test\SAM_1st_manual'
# sam_mask_root = r'E:\Unet\CHASEDB1\test\labels'
# sam_mask_names = os.listdir(sam_mask_root)  # 读取图像的路径
# sam_mask_path = [os.path.join(sam_mask_root, name) for name in sam_mask_names]

# new_mask_root = r'E:\Unet\DRIVE\test\new_1st_manual2'
# new_mask_names = os.listdir(new_mask_root)  # 读取图像的路径
# new_mask_path = [os.path.join(new_mask_root, name) for name in new_mask_names]

gt_mask_root = r'/root/Unet/Unet/DRIVE/test/1st_manual'
# gt_mask_root = r'E:\Unet\CHASEDB1\test\labels'
gt_mask_names = os.listdir(gt_mask_root)  # 读取图像的路径
gt_mask_path = [os.path.join(gt_mask_root, name) for name in gt_mask_names]
gt_mask_path=sorted(gt_mask_path)

total_pixels = 0.
acc = 0.
for i,path in enumerate(imgs_path):
    with torch.no_grad():
        img = Image.open(path)  # 读取预测的图片
        img = transform(img)  # 预处理
        img = torch.unsqueeze(img, dim=0)  # 增加batch维度
        #pred=img
        pred = net(img.to(device))  # 网络预测

        pred = torch.squeeze(pred)  # 将(batch、channel)维度去掉
        pred = np.array(pred.data.cpu())  # 保存图片需要转为cpu处理

        pred[pred >= 0] = 255  # 转为二值图片
        pred[pred < 0] = 0

        # new_size = (960, 999)
        new_size = (584, 565)
        new_mask = np.resize(pred, new_size).astype(np.uint8)

        # sam_mask = Image.open(sam_mask_path[i])
        # sam_mask = np.array(sam_mask)
        # sam_mask = np.resize(sam_mask, new_size).astype(np.uint8)

        # xor_mask = Image.open(new_mask_path[i])
        # xor_mask = np.array(xor_mask)
        # xor_mask = np.resize(xor_mask,new_size).astype(np.uint8)

        gt_mask = Image.open(gt_mask_path[i])
        gt_mask = np.array(gt_mask)
        # gt_mask = np.resize(gt_mask, new_size).astype(np.uint8)

        # xor_mask2 = np.logical_xor(gt_mask,sam_mask)

        # combined_mask = np.logical_or(sam_mask, xor_mask)
        # combined_mask = np.where(combined_mask, 255, 0).astype(np.uint8)

        matching_pixels = np.sum(gt_mask == new_mask)

        # 获取总像素数量（数组形状的乘积）
        total_pixels += gt_mask.size

        # 计算准确率（Accuracy）
        acc += matching_pixels

accuracy = acc / total_pixels
# 打印准确率
print("Accuracy:", accuracy)


