from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import cv2

gt_root = r'E:\Unet\DRIVE\test\1st_manual'
gt_mask_names = os.listdir(gt_root)  # 读取图像的路径
gt_mask_path = [os.path.join(gt_root, name) for name in gt_mask_names]

sam_root = r'E:\Unet\DRIVE\test\SAM_1st_manual'
sam_mask_names = os.listdir(sam_root)
sam_mask_path = [os.path.join(sam_root, name) for name in sam_mask_names]

save_path = r'E:\Unet\DRIVE\test\new_1st_manual'

for i in range(len(sam_mask_path)):
    mask = Image.open(gt_mask_path[i]).convert("L")
    mask = np.asarray(mask, dtype=float)

    sam_mask = Image.open(sam_mask_path[i]).convert("L")
    sam_mask = np.asarray(sam_mask, dtype=float)

    # 计算放缩比例
    # zoom_ratio = (sam_mask.shape[0] / mask.shape[0], sam_mask.shape[1] / mask.shape[1])
    zoom_ratio = (mask.shape[0] / sam_mask.shape[0], mask.shape[1] / sam_mask.shape[1])

    # 调整 mask 的大小以匹配 sam_mask 的形状
    sam_mask = zoom(sam_mask, zoom_ratio, order=1)

    # 将数组数据类型转换为布尔型执行异或操作
    xor_mask = (sam_mask > 0.5) ^ (mask > 0.5)
    xor_mask = xor_mask.astype(float)

    plt.imsave(save_path + '\\' + gt_mask_names[i].split('.')[0] + ".png", xor_mask, cmap='gray')

