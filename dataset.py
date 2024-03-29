import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

data_transform = {
    "train": transforms.Compose([transforms.ToTensor(),
                                 transforms.Resize((1024, 1024)),
                                 transforms.Normalize((0.5,), (0.5,))]),
    "test": transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((1024, 1024)),
                                ])
}


# 数据处理文件
class Data_Loader(Dataset):  # 加载数据
    def __init__(self, root, mask_root,cut,transforms_train=data_transform['train'], transforms_test=data_transform['test']):  # 初始化
        imgs = os.listdir(root)# 读取图像的路径
        self.imgs = [os.path.join(root, img) for img in imgs]# 取出路径下所有的图片
        mask_imgs = os.listdir(mask_root)
        self.mask = [os.path.join(mask_root, img) for img in mask_imgs]
        self.imgs=sorted(self.imgs)
        self.mask=sorted(self.mask)
        self.cut=cut
        if cut!=False:
            self.imgs=self.imgs[:cut]
            self.mask=self.mask[:cut]
        self.transforms_train = transforms_train  # 预处理
        self.transforms_test = transforms_test
        

    def __getitem__(self, index):  # 获取数据、预处理等等
        image_path = self.imgs[index]  # 根据index读取图片
        label_path = self.mask[index]

        image = Image.open(image_path)  # 读取图片和对应的label图
        label = Image.open(label_path)

        image = self.transforms_train(image)  # 样本预处理

        label = self.transforms_test(label)  # labels 预处理
        label[label > 0] = 1

        return image, label

    def __len__(self):  # 返回样本的数量
        return len(self.imgs)