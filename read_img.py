from PIL import Image
import numpy as np
import os

# x = Image.open(r'E:\Unet\DRIVE\train\new_1st_manual2\21_manual1.png')
# x = np.array(x)

root = r'E:\Unet\DRIVE\test\new_1st_manual'
mask_names = os.listdir(root)  # 读取图像的路径
mask_path = [os.path.join(root, name) for name in mask_names]

save_path = r'E:\Unet\DRIVE\test\new_1st_manual2'

for i in range(len(mask_path)):
    img = Image.open(mask_path[i])
    img = img.convert('L')
    img.save(save_path + '\\' + mask_names[i].split('.')[0] + '.png')

