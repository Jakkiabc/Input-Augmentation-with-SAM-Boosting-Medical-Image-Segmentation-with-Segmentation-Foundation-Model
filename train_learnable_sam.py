#coding:utf-8
import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as opt
from PIL import Image
import argparse
import numpy as np
from albumentations import Compose, Resize, Normalize, ColorJitter, HorizontalFlip, VerticalFlip
import glob
import os
import re

import log

parser = argparse.ArgumentParser("Learnable prompt")
parser.add_argument("--image", type=str, required=True, 
                    help="path to the image that used to train the model")
parser.add_argument("--mask_path", type=str, required=True,
                    help="path to the mask file for training")
parser.add_argument("--epoch", type=int, default=32, 
                    help="training epochs")
parser.add_argument("--checkpoint", type=str, required=True,
                    help="path to the checkpoint of sam")
parser.add_argument("--model_name", default="default", type=str,
                    help="name of the sam model, default is vit_h",
                    choices=["default", "vit_b", "vit_l", "vit_h"])
parser.add_argument("--save_path", type=str, default="./ckpt_prompt",
                    help="save the weights of the model")
parser.add_argument("--num_classes", type=int, default=12)
parser.add_argument("--mix_precision", action="store_true", default=False,
                    help="whether use mix precison training")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--optimizer", default="adamw", type=str,
                    help="optimizer used to train the model")
parser.add_argument("--weight_decay", default=5e-4, type=float, 
                    help="weight decay for the optimizer")
parser.add_argument("--momentum", default=0.9, type=float,
                    help="momentum for the sgd")
parser.add_argument("--batch_size", default=1, type=int)
parser.add_argument("--divide", action="store_true", default=False,
                    help="whether divide the mask")
parser.add_argument("--divide_value", type=int, default=255, 
                    help="divided value")
parser.add_argument("--num_workers", "-j", type=int, default=1, 
                    help="divided value")
parser.add_argument("--device", default="0", type=str)
parser.add_argument("--model_type", default="sam", choices=["dino", "sam"], type=str,
                    help="backbone type")
parser.add_argument("--log_file_name",type=str,default="log/log.txt")
parser.add_argument("--pth_name",type=str,default="default")
parser.add_argument("--save_image_name",type=str,default="output/output.png")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.device

from learnerable_seg import PromptSAM, PromptDiNo
from scheduler import PolyLRScheduler
from metrics.metric import Metric

class SegDataset:
    def __init__(self, img_paths, mask_paths, 
                 mask_divide=False, divide_value=255,
                 pixel_mean=[0.5]*3, pixel_std=[0.5]*3,
                 img_size=(584,565)) -> None:
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.length = len(img_paths)
        self.mask_divide = mask_divide
        self.divide_value = divide_value
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std
        self.img_size = img_size
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        mask_path = self.mask_paths[index]
        img = Image.open(img_path).convert("RGB")
        img = np.asarray(img)
        mask = Image.open(mask_path).convert("L")
        mask = np.asarray(mask)
        mask[mask==255] = 1
        #print(1,mask.shape)
        #print(2,img.shape)
        if self.mask_divide:
            mask = mask // self.divide_value
        transform = Compose(
            [
                ColorJitter(),
                VerticalFlip(),
                HorizontalFlip(),
                Resize(self.img_size, self.img_size),
                Normalize(mean=self.pixel_mean, std=self.pixel_std)
            ]
        )
        aug_data = transform(image=img, mask=mask)
        x = aug_data["image"]
        target = aug_data["mask"]
        if img.ndim == 3:
            x = np.transpose(x, axes=[2, 0, 1])
        elif img.ndim == 2:
            x = np.expand_dims(x, axis=0)
        return torch.from_numpy(x), torch.from_numpy(target)
    def net_to_onnx(net, onnx_path, input_shapes, input_names, output_names, dynamic_batch_size=False):
        inputs = tuple([torch.rand(x) for x in input_shapes])
        dynamic_axes = {k: {0: 'batch_size'} for k in input_names + output_names} if dynamic_batch_size else None
        os.makedirs(osp.dirname(osp.realpath(onnx_path)), exist_ok=True)
        net.eval()
        torch.onnx.export(net, inputs, onnx_path, verbose=True, input_names=input_names, output_names=output_names,
                      opset_version=13, dynamic_axes=dynamic_axes)

        print(f'pytorch to onnx successful | save | {onnx_path}')

def main(args):
    img_path = args.image
    mask_path = args.mask_path
    epochs = args.epoch
    checkpoint = args.checkpoint
    model_name = args.model_name
    save_path = args.save_path
    optimizer = args.optimizer
    weight_decay = args.weight_decay
    lr = args.lr
    momentum = args.momentum
    bs = args.batch_size
    divide = args.divide
    divide_value = args.divide_value
    num_workers = args.num_workers
    model_type = args.model_type
    pth_name=args.pth_name
    # pixel_mean=[123.675, 116.28, 103.53],
    # pixel_std=[58.395, 57.12, 57.375],
    # pixel_mean = np.array(pixel_mean) / 255
    # pixel_std = np.array(pixel_std) / 255
    pixel_mean = [0.5]*3
    pixel_std = [0.5]*3
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    num_classes = args.num_classes
    basename = os.path.basename(img_path)
    _, ext = os.path.splitext(basename)
    if ext == "":
        regex = re.compile(".*\.(jpe?g|png|gif|tif|bmp)$", re.IGNORECASE)
        img_paths = [file for file in glob.glob(os.path.join(img_path, "*.*")) if regex.match(file)]
        print("train with {} imgs".format(len(img_paths)))
        mask_paths = [os.path.join(mask_path, os.path.basename(file)) for file in img_paths]
    else:
        bs = 1
        img_paths = [img_path]
        mask_paths = [mask_path]
        num_workers = 1
    img_size = 1024
    #img_size=(584,565)
    if model_type == "sam":
        model = PromptSAM(model_name, checkpoint=checkpoint, num_classes=num_classes, reduction=4, upsample_times=2, groups=4)
    elif model_type == "dino":
        model = PromptDiNo(name=model_name, checkpoint=checkpoint, num_classes=num_classes)
        img_size = 518
    dataset = SegDataset(img_paths, mask_paths=mask_paths, mask_divide=divide, divide_value=divide_value,
                         pixel_mean=pixel_mean, pixel_std=pixel_std, img_size=img_size)
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=num_workers)
    scaler = torch.cuda.amp.grad_scaler.GradScaler()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    if optimizer == "adamw":
        optim = opt.AdamW([{"params":model.parameters(), "initia_lr": lr}], lr=lr, weight_decay=weight_decay)
    elif optimizer == "sgd":
        optim = opt.SGD([{"params":model.parameters(), "initia_lr": lr}], lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=True)
    loss_func = nn.CrossEntropyLoss()
    scheduler = PolyLRScheduler(optim, num_images=len(img_paths), batch_size=bs, epochs=epochs)
    metric = Metric(num_classes=num_classes)
    best_iou = 0.
    for epoch in range(epochs):
        for i, (x, target) in enumerate(dataloader):
            x = x.to(device)
            target = target.to(device, dtype=torch.long)
            optim.zero_grad()
            if device_type == "cuda" and args.mix_precision:
                x = x.to(dtype=torch.float16)
                with torch.autocast(device_type=device_type, dtype=torch.float16):
                    pred = model(x)
                    #print(1,pred)
                    #print(2,target)
                    loss = loss_func(pred, target)

                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                x = x.to(dtype=torch.float32)
                pred = model(x)
                #print(1,pred.shape)
                #print(2,target.shape)
                loss = loss_func(pred, target)
                loss.backward()
                optim.step()
            metric.update(torch.softmax(pred, dim=1), target)
            #print("epoch:{}-{}: loss:{}".format(epoch+1, i+1, loss.item()))
            logger.info("epoch:{}-{}: loss:{}".format(epoch+1, i+1, loss.item()))
            scheduler.step()
        iou = np.nanmean(metric.evaluate()["iou"][1:].numpy())
        acc = np.nanmean(metric.evaluate()["acc"][1:].numpy())
        #auc = np.nanmean(metric.evaluate()["auc"][1:].numpy())
        f1 = np.nanmean(metric.evaluate()["f1"][1:].numpy())
        recall = np.nanmean(metric.evaluate()["recall"][1:].numpy())
        #shape=pred.shape
        #logger.info("epoch-{}: shape:{}".format(epoch, shape.item()))
        #print("epoch-{}: iou:{}".format(epoch, iou.item()))
        logger.info("epoch-{}: iou:{}".format(epoch, iou.item()))
        logger.info("epoch-{}: acc:{}".format(epoch, acc.item()))
        #logger.info("epoch-{}: auc:{}".format(epoch, auc.item()))
        logger.info("epoch-{}: f1:{}".format(epoch, f1.item()))
        logger.info("epoch-{}: recall:{}".format(epoch, recall.item()))
        if iou > best_iou:
            best_iou = iou
            torch.save(
                model.state_dict(), os.path.join(save_path, "{}_{}_prompt_{}.pth".format(model_type, model_name,pth_name))
            )
    return pred

if __name__ == "__main__":
    import torch,logging
    logger = log.Logger(
        log_file_name=args.log_file_name,
        log_level=logging.DEBUG,
        logger_name="trainlog",
    ).get_log()
    pred=main(args)
    #print(pred)
    #print(pred.shape)
    # 转换张量为PIL图像
    from PIL import Image
    #pred = pred.squeeze().permute(1, 2, 0).cpu().numpy().astype('uint8')
    pred = pred.argmax(1).squeeze()
    #print(pred.shape)
    #pred = pred.squeeze().permute(1, 2, 0)  # 调整张量维度
    pred = (pred - pred.min()) / (pred.max() - pred.min())  # 将张量值归一化到0-1范围
    pred = (pred * 255).to(torch.uint8)  # 将值缩放到0-255范围，并转换为整数类型
    image = Image.fromarray(pred.cpu().numpy())  # 创建PIL图像
    imagepath=args.save_image_name

# 保存图像或显示图像
    image.save(imagepath)  # 保存图像为PNG文件
    #image.show()  # 显示图像
    #import cv2
    #cv2.imwrite("modeltest.png",model)
    #segment_anything="segmeng-anything/segment_anything"

    #from segment_anything import predictor,SamPredictor
    
    #pred = SamPredictor(model)
    #image="/root/sam/LearnablePromptSAM-main/im0001.png"
    #predictor.set_image(image)
    #masks, _, _ = pred.predict(
        
    #point_coords=None,
    #input_box=None,
    #box=input_box[None, :],
    #multimask_output=False,
#)
    #mask=masks[0]
#plt.imshow(mask)
    #from PIL import Image
    #mask_image = Image.fromarray(( * 255).astype(np.uint8))
    #mask_image.save('masktest.png')
    
    #point_labels=None,
