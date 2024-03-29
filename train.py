from model import UNet
from dataset import Data_Loader
from torch import optim
import torch.nn as nn
import torch
from tqdm import tqdm
import log
from torch.utils.data import ConcatDataset
import torch,logging
logger = log.Logger(
        log_file_name='/root/Unet/Unet/logtxtALL_AUG_trainlog_2.txt',
        log_level=logging.DEBUG,
        logger_name="ALL_AUG_trainlog",
        ).get_log()

# 网络训练模块
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # GPU or CPU
print(device)
net = UNet(in_channels=3, num_classes=1)  # 加载网络
net.to(device)  # 将网络加载到device上

# 加载训练集
trainset = Data_Loader(root=r"/root/Unet/Unet/DRIVE/train/Aug_images",
                     mask_root=r'/root/Unet/Unet/DRIVE/train/mask',cut=5)
trainset2 = Data_Loader(root=r"/root/Unet/Unet/STARE/train/output/Aug",
                       mask_root=r'/root/Unet/Unet/STARE/train/labels',cut=3)
trainset3 = Data_Loader(root=r"/root/Unet/Unet/CHASEDB1/train/SAMAug_train_CHASEDB1_images",
                       mask_root=r'/root/Unet/Unet/CHASEDB1/train/labels',cut=2)

train_loader=torch.utils.data.DataLoader(dataset=ConcatDataset([trainset,trainset2,trainset3]),batch_size=1,shuffle=True)
#train_loader2 = torch.utils.data.DataLoader(dataset=trainset2, batch_size=1,shuffle=True)
#train_loader3 = torch.utils.data.DataLoader(dataset=trainset3, batch_size=1,shuffle=True)
#combined_dataloader = DataLoader(ConcatDataset([dataloader1.dataset, dataloader2.dataset]), batch_size=32, shuffle=True)
  # 样本总数为 31
len=10
# 加载测试集
testset = Data_Loader(root=r"/root/Unet/Unet/DRIVE/test/images", mask_root=r"/root/Unet/Unet/DRIVE/test/mask",cut=2)
testset2 = Data_Loader(root=r"/root/Unet/Unet/STARE/test/output/Aug", mask_root=r"/root/Unet/Unet/STARE/test/labels",cut=False)
testset3 = Data_Loader(root=r"/root/Unet/Unet/CHASEDB1/test/images", mask_root=r"/root/Unet/Unet/CHASEDB1/test/labels",cut=False)
test_loader = torch.utils.data.DataLoader(dataset=ConcatDataset([testset,testset2,testset3]), batch_size=1,shuffle=False)

# 加载优化器和损失函数
# optimizer = optim.RMSprop(net.parameters(), lr=0.00001, weight_decay=1e-8, momentum=0.9)  # 定义优化器
optimizer = optim.Adam(net.parameters(),lr=5e-4)
criterion = nn.BCEWithLogitsLoss()  # 定义损失函数

# 保存网络参数
save_path = '/root/Unet/Unet/pth/one-UNet_DRIVE_AUG_All_2.pth'  # 网络参数的保存路径
best_acc = 0.0  # 保存最好的准确率

# 训练
for epoch in range(50):

    net.train()  # 训练模式
    running_loss = 0.0

    for image, label in tqdm(train_loader):
        optimizer.zero_grad()  # 梯度清零
        pred = net(image.to(device))  # 前向传播
        loss = criterion(pred, label.to(device))  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 梯度下降

        running_loss += loss.item()  # 计算损失和

    net.eval()  # 测试模式
    acc = 0.0  # 正确率
    total = 0
    with torch.no_grad():
        for test_image, test_label in tqdm(test_loader):
            outputs = net(test_image.to(device))  # 前向传播

            outputs[outputs >= 0] = 1  # 将预测图片转为二值图片
            outputs[outputs < 0] = 0

            # 计算预测图片与真实图片像素点一致的精度：acc = 相同的 / 总个数
            acc += (outputs == test_label.to(device)).sum().item() / (test_label.size(2) * test_label.size(3))
            total += test_label.size(0)

    accurate = acc / total  # 计算整个test上面的正确率
    train_loss=running_loss/len
    print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f %%' %
          (epoch + 1, running_loss / len, accurate * 100))
    logger.info("epoch:{} ,train_loss:{},test_accuracy:{}".format(epoch+1,  train_loss,accurate))

    if accurate > best_acc:  # 保留最好的精度
        best_acc = accurate
        
        torch.save(net.state_dict(), save_path)  # 保存网络参数

print(f'best acc:{best_acc}')
logger.info(" best_acc:{}".format( best_acc))

