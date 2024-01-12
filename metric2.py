import cv2
import numpy as np
import os
from PIL import Image
from sklearn.metrics import roc_curve,auc

#预测结果路径
pred_path = r'E:\Unet\DRIVE\test\SAM_1st_manual'
#标签路径
lab_path = r'E:\Unet\DRIVE\test\1st_manual'


def tpcount(imgp,imgl):
    n = 0
    for i in range(WIDTH):
        for j in range(HIGTH):
            if imgp[i,j] == 255 and imgl[i,j] == 255:
                n = n+1
    return n

def fncount (imgp,imgl):
    n = 0
    for i in range(WIDTH):
        for j in range(HIGTH):
            if imgl[i,j] == 255 and imgp[i,j] == 0:
                n = n+1
    return n

def fpcount(imgp,imgl):
    n = 0
    for i in range(WIDTH):
        for j in range(HIGTH):
            if imgl[i,j] == 0 and imgp[i,j] == 255:
                n+=1
    return n

def tncount(imgp,imgl):
    n=0
    for i in range(WIDTH):
        for j in range(HIGTH):
            if imgl[i,j] == 0 and imgp[i,j] == 0:
                n += 1
    return n




imgs = os.listdir(pred_path)
a = len(imgs)
TP = 0
FN = 0
FP = 0
TN = 0
c = 0

root = r'E:\Unet\DRIVE\test\SAM_1st_manual'
imgs_names = os.listdir(root)  # 读取图像的路径
imgs_path = [os.path.join(root, name) for name in imgs_names]

gt_mask_root = r'E:\Unet\DRIVE\test\1st_manual'
gt_mask_names = os.listdir(gt_mask_root)  # 读取图像的路径
gt_mask_path = [os.path.join(gt_mask_root, name) for name in gt_mask_names]

for i in range(len(imgs_path)):

    imgp = Image.open(imgs_path[i])
    imgp = np.array(imgp)

    new_size = (584, 565)
    imgp = np.resize(imgp, new_size).astype(np.uint8)

    imgl = Image.open(gt_mask_path[i])
    imgl = np.array(imgl)

    WIDTH = imgl.shape[0]
    HIGTH = imgl.shape[1]

    TP += tpcount(imgp, imgl)
    FN += fncount(imgp, imgl)
    FP += fpcount(imgp, imgl)
    TN += tncount(imgp, imgl)

    c += 1
    print('已经计算：'+str(c) + ',剩余数目：'+str(a-c))

print('TP:'+str(TP))
print('FN:'+str(FN))
print('FP:'+str(FP))
print('TN:'+str(TN))


#准确率
acc = (int(TN)+int(TP))/(int(WIDTH)*int(HIGTH)*int(len(imgs)))
#精确率
precision = int(TP)/(int(TP)+int(FP))
#召回率
recall = int(TP)/(int(TP)+int(FN))
#F1
f1 = int(TP)*2/(int(TP)*2+int(FN)+int(FP))
# FPR
FPR = int(FP)/(int(FP)+int(TN))
# TPR
TPR = int(TP)/(int(TP)+int(FN))


print('acc：'+ str(acc))
print('precision：'+ str(precision))
print('recall：'+ str(recall))
print('F1值：'+ str(f1))

