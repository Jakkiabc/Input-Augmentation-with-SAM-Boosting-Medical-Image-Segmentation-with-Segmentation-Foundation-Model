# coding: utf-8
# created by qiutianyu on 20221201
import os
from os import path as osp
import numpy as np
import onnxruntime as ort
import cv2
import skimage

dir_path = osp.dirname(osp.abspath(__file__))

providers = [
    # ('TensorrtExecutionProvider', {
    #     'device_id': 0,
    #     'trt_max_workspace_size': 1024 * 1024 * 1024,  # 1GB
    #     'trt_fp16_enable': True,
    #     'trt_engine_cache_enable': True,
    #     'trt_engine_cache_path': osp.join(dir_path, 'trt_cache'),
    # }),
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'gpu_mem_limit': 1024 * 1024 * 1024,  # 1GB
        'cudnn_conv_algo_search': 'EXHAUSTIVE',
        'do_copy_in_default_stream': True,
    }),
    'CPUExecutionProvider',
]


def process_inputs(image_path, h=1024, w=1024):
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
    # image = cv2.imread(image_path)
    pixel_mean = [0.5] * 3
    pixel_std = [0.5] * 3
    # 计算图像当前的均值和标准差
    image_float = image.astype(np.float32)
    target_mean = 0.5  # 例如，将均值调整为0.5
    target_std = 0.5  # 例如，将标准差调整为0.2

    current_mean = np.mean(image_float)
    current_std = np.std(image_float)

    # 使用线性变换来调整均值和标准差
    adjusted_image = (image_float - current_mean) * (target_std / current_std) + target_mean

    # 将浮点数图像转换回整数类型
    # adjusted_image = np.clip(adjusted_image, 0, 255).astype(np.uint8)

    inputs = cv2.dnn.blobFromImage(adjusted_image, size=(w, h), swapRB=False)

    return inputs


def read_txt(txt_path):
    path_list = []
    labels_list = []
    with open(txt_path, encoding='utf-8') as f:
        for line in f:
            path = line.rstrip('\n').split(',')[0]
            labels = line.rstrip('\n').split(',')[1:]
            labels = [int(x) for x in labels]
            path_list.append(path)
            labels_list.append(labels)
    return path_list, labels_list


def softmax(x):
    x_max = np.max(x)
    t_exp = np.exp(x - x_max)
    t_sum = np.sum(t_exp)
    return t_exp / t_sum


def inference(onnx_path, img_path):
    print('init onnxruntime session')
    sess = ort.InferenceSession(onnx_path, providers=providers)
    inputs = process_inputs(image_path=img_path)
    outputs = sess.run(None, {'nchw_bgr': inputs})
    return outputs


if __name__ == '__main__':
    from PIL import Image

    #onnx_path = osp.join(dir_path, 'E:\LearnablePromptSAM-main\ckpts\sam_21_l2000_chase5.onnx')
    onnx_path='/root/sam/LearnablePromptSAM-main/pretrained/sam_21_l2000.onnx'
    save_path = r'/root/Unet/Unet/STARE/train/output/sam_mask/'
    root = r'/root/Unet/Unet/STARE/train/images'
    imgs_names = os.listdir(root)  # 读取图像的路径
    all_imgs_path = [os.path.join(root, img) for img in imgs_names]  # 取出路径下所有的图片

    for i, img_path in enumerate(all_imgs_path):
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)
        img = skimage.img_as_float(img)

        pred = inference(onnx_path, img_path)

        pred = pred[0]

        pred = pred.argmax(1).squeeze()
        pred = (pred - pred.min()) / (pred.max() - pred.min())  # 将张量值归一化到-1范围
        pred = (pred * 255).astype(np.uint8)

        # new_size = (584, 565)
        # new_size = (960, 999)
        # new_mask = np.resize(pred, new_size).astype(np.uint8)

        # img[:, :, 2] = img[:, :, 2] + new_mask
        # img = (img * 255).astype(np.uint8)
        # img[img >= 255] = 1
        # image = Image.fromarray(img)

        pred[pred >= 255] = 255
        image = Image.fromarray(pred,mode='L')

        name = imgs_names[i].split('.')[0]
        image.save(save_path  + name + '.png')

        print(name+'\nfinished!')
