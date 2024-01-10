import cv2
import numpy as np
import torch
from PIL import Image
import skimage
import os
from skimage.segmentation import find_boundaries
def SAMAug(tI, pred,ss):
    #masks = mask_generator.generate(tI)
    tI = skimage.img_as_float(tI)
    pred = skimage.img_as_float(pred)
    #print(1,tI.shape)
    SegPrior = np.zeros((tI.shape[0], tI.shape[1])) #构建原图大小的shape
   # print(1,SegPrior.shape)
    BoundaryPrior = np.zeros((tI.shape[0], tI.shape[1]))
    thismask=pred
    thismask_ = np.zeros((thismask.shape))
    thismask_[np.where(thismask == 255)] = 1
    #print(thismask_.shape)
    #index=np.where(thismask==255)
    #print(np.where(thismask_==1))
    SegPrior[np.where(thismask_ == 1)] = SegPrior[np.where(thismask_ == 1)] + ss
    BoundaryPrior = BoundaryPrior + find_boundaries(thismask_, mode='thick')
    BoundaryPrior[np.where(BoundaryPrior > 0)] = 1
    tI[:, :, 1] = tI[:, :, 1] + SegPrior
    tI[:, :, 2] = tI[:, :, 2] + BoundaryPrior
    return BoundaryPrior,SegPrior,tI

def calculate_stability_score(
    masks, mask_threshold, threshold_offset
) :
    """
    Computes the stability score for a batch of masks. The stability
    score is the IoU between the binary masks obtained by thresholding
    the predicted mask logits at high and low values.
    """
    # One mask is always contained inside the other.
    # Save memory by preventing unnecessary cast to torch.int64
    intersections = (
        (masks > (mask_threshold + threshold_offset))
        .sum(-1, dtype=torch.int16)
        .sum(-1, dtype=torch.int32)
    )
    unions = (
        (masks > (mask_threshold - threshold_offset))
        .sum(-1, dtype=torch.int16)
        .sum(-1, dtype=torch.int32)
    )
    return intersections / unions


if __name__ == '__main__':
#     list=open('/root/sam/LearnablePromptSAM-main/txt/realtestpiliang.txt')
#     txt=list.readlines()
#     reallist=[]
#     for w in txt:
#         w=w.replace('\n','')
#         reallist.append(w)
# #print(a)
# #print(list.readlines())
#     list.close()
#     list_2=open('/root/sam/LearnablePromptSAM-main/txt/masktestpiliang.txt')
#     txt_2=list_2.readlines()
# #     masklist=[]
#     for a in txt_2:
#         a=a.replace('\n','')
#         masklist.append(a)
# #print(a)
# #print(list.readlines())
#     list.close()
    #newfilepath="{}.txt"
    root = r'/root/Unet/Unet/data/CHASE/train/origin'
    mask_names = os.listdir(root)  # 读取图像的路径
    reallist = [os.path.join(root, name) for name in mask_names]
    reallist=sorted(reallist)
    print(reallist)
    root2 = r'/root/Unet/Unet/data/CHASE/train/mask_1'
    mask_names2 = os.listdir(root2)  # 读取图像的路径
    masklist = [os.path.join(root2, name) for name in mask_names2]
    masklist=sorted(masklist)
    print(masklist)
    
    
    
    for i in range(len(reallist)):
        #file_path = v
        print("beginng:{}".format(i+21))
        realimagepath=reallist[i]
        realimage=cv2.imread(realimagepath)
        realimage_rgb = cv2.cvtColor(realimage, cv2.COLOR_BGR2RGB)
        realimage_rgb=cv2.resize(realimage_rgb,(1024,1024))
        imagepath=masklist[i]
        image = Image.open(imagepath)
        image = np.array(image)
        image=image.astype(np.float32)
        torch_pr = torch.from_numpy(image)
        score=calculate_stability_score(
    torch_pr, mask_threshold=0.0, threshold_offset=1.0 )  
        score=score.tolist()
        BoundaryPrior,SegPrior,SAMAu=SAMAug(realimage_rgb,image,score)
        SAMAu = (SAMAu - SAMAu.min()) / (SAMAu.max() - SAMAu.min())
        BoundaryPrior = (BoundaryPrior - BoundaryPrior.min()) / (BoundaryPrior.max() - BoundaryPrior.min())
        SegPrior == (SegPrior - SegPrior.min()) / (SegPrior.max() - SegPrior.min())
        BoundaryPrior = (BoundaryPrior * 255).astype(np.uint8)
        SegPrior = (SegPrior * 255).astype(np.uint8)
        SAMAu = (SAMAu * 255).astype(np.uint8)
        Bound = Image.fromarray(BoundaryPrior)
        Seg=Image.fromarray(SegPrior)
        Aug=Image.fromarray(SAMAu)
        Boundpath='/root/Unet/Unet/data/CHASE/train/output/Bound_1/Bound_{}.png'.format(i)
        Segpath='/root/Unet/Unet/data/CHASE/train/output/Seg_1/Seg_{}.png'.format(i)
        Augpath='/root/Unet/Unet/data/CHASE/train/output/Aug_1/Aug_{}.png'.format(i)
        Bound.save(Boundpath)
        Seg.save(Segpath)
        Aug.save(Augpath)
        print("finished！")
        del BoundaryPrior,SegPrior,SAMAu,Bound,Seg,Aug
    
                    