# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 01:10:13 2020

@author: YifanYang
"""

# 调用必要的函数包
import os
# import gc
import numpy as np
import matplotlib.pyplot as plt
# from collections import namedtuple

# import random
import torch
# from torch.utils import data
from torch import nn
from torch import optim
# from torch import autograd
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")
# from memory_profiler import profile


# 定义全局变量
SIZE=200
DATASET_NUM = 80000/SIZE
PATH='.\\0数据集\\TSP\\TSP_N6M1P6\\'   #数据保存位置目录名称
F_IN = PATH+'TSP_N6M1P6_In'    #input地址
F_OUT = PATH+'TSP_N6M1P6_Out'  #label地址
DIR='.\\tsp_1\\' #储存结果的文件夹
DIMENSION_STATE=448 # 图像的大小
BATCH_SIZE=5 #读取数据的批大小
CLASS_NUM=2 #输出像素类型数量
LEARNING_RATE=3E-4 # 学习率(e-3~e-5之间，3倍为一档)
momentum = 0.5 # 优化算法的超参数（0.5~0.9之间）

# 函数：从不同的文件中读取数据
def read_data(signal):
    X_dir=F_IN+str(signal+1)+".npy"
    Y_dir=F_OUT+str(signal+1)+".npy"
    X_train=np.load(X_dir)
    Y_train=np.load(Y_dir)
    return X_train,Y_train

# 从文件中提取数据
# X_train=np.load(F_IN +"1.npy") # 训练集的输入
# Y_train=np.load(F_OUT +"1.npy") # 训练集的输出
X_test=np.load(F_IN+"400.npy") # 测试集的输入
Y_test=np.load(F_OUT+"400.npy") # 测试集的输出

# ### 调试： 数据大小展示
# print(X_train.shape)
# print(Y_train.shape) 

# 函数：从读取的数据中随机获取部分样本进行训练
def random_batch(X_train,Y_train,batch_size):
    rnd_indices = np.random.randint(0, len(X_train), batch_size) # 生成随机数（组）
    X_batch = X_train[rnd_indices]
    X_batch=torch.from_numpy(X_batch)# 转换成tensor
    X_batch=X_batch.type(torch.FloatTensor) # 将tensor统一为float64格式  #数据类型对训练速度没有太大影响
    Y_batch = Y_train[rnd_indices]
    Y_batch=torch.from_numpy(Y_batch)
    Y_batch=Y_batch.type(torch.FloatTensor)
#     %xdel 
    return X_batch, Y_batch

# 函数：从读取的数据中定向获取部分样本进行训练
def direct_batch(X_test,Y_test,org,size):
    if org+size>SIZE:
        org=SIZE-size
    X_batch = X_test[org:(org+size)]
    X_batch=torch.from_numpy(X_batch)# 转换成tensor
    X_batch=X_batch.type(torch.FloatTensor) # 将tensor统一为float64格式  #数据类型对训练速度没有太大影响
    Y_batch = Y_test[org:(org+size)]
    Y_batch=torch.tensor(Y_batch)
    return X_batch, Y_batch

# ### 调试： 函数功能展示
# X_batch,Y_batch=random_batch(X_train,Y_train,BATCH_SIZE)
# print(X_batch.shape)
# print(Y_batch.shape)

# ### 调试： 展示输入输出
# case_number=0
# plt.figure()
# plt.subplot(121)
# plt.imshow(X_test[case_number,:,:])
# plt.subplot(122)
# plt.imshow(Y_test[case_number,:,:])

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),  #归一化层
            nn.ReLU(inplace=True),  # 激活函数层
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        out = self.conv(x)
        return out
    
# 定义深度学习网络类
class Unet(nn.Module):
    # 函数：初始化
    def __init__(self, in_ch, out_ch):
        super(Unet, self).__init__()
        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = DoubleConv(512, 1024)
        # 逆卷积，也可以使用上采样(保证k=stride,stride即上采样倍数)
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)

        return c10
    
# 检测是否有可用的GPU，有则使用，否则使用CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)



# 实例化网络
#model = Net()
model=Unet(1,CLASS_NUM)  # Unet(in_channel,out_channel)
model.to(device)

# 定义损失函数
# criterion = nn.CrossEntropyLoss()
# 带权重
p_w=np.array([1,5])
P_W=torch.from_numpy(p_w).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=P_W)
# # 不带权重
# criterion = nn.BCEWithLogitsLoss()

# 定义优化器
# optimizer = optim.SGD(model.parameters(),lr=LEARNING_RATE,momentum=momentum)
optimizer = optim.Adam(model.parameters(),lr=LEARNING_RATE)

####################################################  断续训练
# 优化器参数提取
state=torch.load(os.path.join(DIR,"optimizer_state_dict.pkl"))
optimizer.load_state_dict(state)
# optimizer = torch.load(os.path.join(DIR,"optimizer_state_dict.pkl"))

# 网络参数提取
# model=torch.load(os.path.join(DIR,'FCN_Unet_model_5th.pkl')).to(device)
state=torch.load(os.path.join(DIR,"FCN_Unet_model_5th.pkl"))
model.load_state_dict(state)
####################################################  断续训练

# @profile
def plot_3_figs(pred,x,y,n):
    plt.subplot(131)
    plt.imshow(x[n,:,:,0],cmap=plt.cm.gray)  #改为0，否则448*448*1不符合热力图格式
    plt.title('input')
    plt.axis('off')  #去掉坐标轴
    plt.subplot(132)
    plt.imshow(pred[n,:,:],cmap=plt.cm.gray)
    plt.title('output')
    plt.axis('off')  #去掉坐标轴
    plt.subplot(133)
    plt.imshow(y[n,:,:],cmap=plt.cm.gray)
    plt.title('label')
    plt.axis('off')  #去掉坐标轴
    
  
pre=130000
loss_t=100
view_t=1000
out_t=1000
switch_t=round(SIZE/(2*BATCH_SIZE))
train_loss=0
train_acc=0
eval_loss = 0
eval_acc = 0
# X_view_batch,Y_view_batch=direct_batch(X_test,Y_test,160,BATCH_SIZE)
TRAIN_LOSS=np.zeros([], dtype = float)
TRAIN_ACC=np.zeros([], dtype = float)
EVAL_LOSS=np.zeros([], dtype = float)
EVAL_ACC=np.zeros([], dtype = float)
OUT=torch.Tensor([])
# 目录准备
if not os.path.exists(DIR+'out\\'):
    os.makedirs(DIR+'out\\')
for i in range(BATCH_SIZE):
    if not os.path.exists(DIR+'lr='+str(LEARNING_RATE)+'\\Validation_'+str(i)):
        os.makedirs(DIR+'lr='+str(LEARNING_RATE)+'\\Validation_'+str(i))
if not os.path.exists(DIR+'lr='+str(LEARNING_RATE)+'\\Train'):
    os.makedirs(DIR+'lr='+str(LEARNING_RATE)+'\\Train')
# 训练集准备
X_train,Y_train=read_data(np.random.randint(DATASET_NUM)+1) 
# 验证集准备
X_test_batch,Y_test_batch=direct_batch(X_test,Y_test,125,BATCH_SIZE) 
X_test_batch = X_test_batch.to(device)
Y_test_batch = Y_test_batch.to(device)
X_test_batch = X_test_batch.unsqueeze(1) 
oh_Y_test_batch=torch.nn.functional.one_hot(Y_test_batch.long(),CLASS_NUM)
# oh_Y_test_batch=oh_Y_test_batch.permute(0,3,1,2)
oh_Y_test_batch=oh_Y_test_batch.float()


for times in range(50000):
    # *********************************进入训练模式*************************************
    model.train()
    # 读取数据
    X_batch,Y_batch=random_batch(X_train,Y_train,BATCH_SIZE)
    X_batch = X_batch.to(device)
    Y_batch = Y_batch.to(device)
    X_batch = X_batch.unsqueeze(1) #[5,1,448,448]
    oh_Y_batch=torch.nn.functional.one_hot(Y_batch.long(),CLASS_NUM) #[5,448,448,2]
#     oh_Y_batch=oh_Y_batch.permute(0,3,1,2)
    oh_Y_batch=oh_Y_batch.float()
    # 前向传播
    out = model(X_batch) #[5,2,448,448]
    out_p=out.permute(0,2,3,1)
    loss = criterion(out_p,oh_Y_batch)
    # 反向传播
    optimizer.zero_grad()#前期梯度清零
    loss.backward()
    optimizer.step()
    # 记录误差
    train_loss += loss.item()/BATCH_SIZE
    # 计算分类的准确率
    _,train_pred = out.max(1)#取输出的最大值
    ####################### 改动 start
    # num_correct = (train_pred==Y_batch.long()).sum().item()
    # acc = num_correct /(BATCH_SIZE*DIMENSION_STATE*DIMENSION_STATE)
    Y_flag = Y_batch.long()   #原Y_batch中，路径为2，long类型除以2后，路径为1，其余为0
    num_all = Y_flag.sum().item()  # 类别1为路径，加和为正确的路径像素数量
    pred_flag = train_pred.long()
    num_correct = (pred_flag.mul(Y_flag)).sum().item()  #两张量内积，仅该像素在Y为1且Pred为1时计数
    acc = num_correct / num_all
    ####################### 改动 end
    train_acc += acc/loss_t
    # *********************************进入测试模式*************************************
    model.eval()
    # 读取数据
#     X_test_batch,Y_test_batch=direct_batch(X_test,Y_test,331,BATCH_SIZE)
#     X_test_batch,Y_test_batch=random_batch(X_test,Y_test,BATCH_SIZE)
#     X_test_batch = X_test_batch.to(device)
#     Y_test_batch = Y_test_batch.to(device)
#     X_test_batch = X_test_batch.unsqueeze(1)
#     oh_Y_test_batch=torch.nn.functional.one_hot(Y_test_batch.long(),CLASS_NUM)
# #     oh_Y_test_batch=oh_Y_test_batch.permute(0,3,1,2)
#     oh_Y_test_batch=oh_Y_test_batch.float()
    # 前向传播
    out = model(X_test_batch)
    out_p=out.permute(0,2,3,1)
    loss = criterion(out_p,oh_Y_test_batch)
    # 记录误差
    eval_loss += loss.item()/BATCH_SIZE
    # 记录测试的准确率
    _,eval_pred = out.max(1)
    ####################### 改动 start
    # num_correct = (eval_pred==Y_test_batch.long()).sum().item()
    # acc = num_correct/(BATCH_SIZE*DIMENSION_STATE*DIMENSION_STATE)
    Y_flag = Y_test_batch.long()   # 原Y_test_batch中，路径为2，long类型除以2后，路径为1，其余为0
    num_all = Y_flag.sum().item()  # 类别1为路径，加和为正确的路径像素数量
    pred_flag = eval_pred.long() 
    num_correct = (pred_flag.mul(Y_flag)).sum().item()  # 两张量内积，仅该像素在Y为1且Pred为1时计数
    acc = num_correct / num_all
    ####################### 改动 end
    eval_acc += acc/loss_t
    # *********************************  记录 *************************************
    # 展示一次准确率
    if (times + pre)%loss_t==loss_t-1: 
        print("迭代：",times+pre+1,"次")
        print("训练集Loss值为：%.05f" %train_loss,end='  ')
        print("训练集正确率为：%.04f" %train_acc,end=' | ')
        print("测试集Loss值为：%.05f" %eval_loss,end='  ')
        print("测试集正确率为：%.04f" %eval_acc)
        print("------------------------------------------------------")
        TRAIN_LOSS=np.append(TRAIN_LOSS,train_loss)# 保存训练误差
        TRAIN_ACC=np.append(TRAIN_ACC,train_acc)# 保存训练正确率
        EVAL_LOSS=np.append(EVAL_LOSS,eval_loss)# 保存测试误差
        EVAL_ACC=np.append(EVAL_ACC,eval_acc)# 保存测试正确率
        train_loss=0
        train_acc=0
        eval_loss = 0
        eval_acc = 0
#     打印一次输出并保存
    if (times + pre)%view_t == 10 and times!=0:
        # 展示验证集情况
        pred_1=eval_pred.cpu()
        pred_2=pred_1.numpy()
        XX=X_test_batch.permute(0,2,3,1)
        X=XX.cpu()
        x=X.numpy().astype(int)
        Y=Y_test_batch.cpu()
        y=Y.numpy()
        # fig=plt.figure()
        # for i in range(BATCH_SIZE):
#
        for i in range(3): ###############输出3个验证集数据的训练过程
            fig=plt.figure()
            plot_3_figs(pred_2,x,y,i)
            # 保存
            plt.savefig(DIR+'lr='+str(LEARNING_RATE)+'\\Validation_'+str(i)+'\\time='+str(times + pre)+'.png')
            plt.close('all')
#
        # 展示训练集训练情况
        pred_1=train_pred.cpu()
        pred_2=pred_1.numpy()
        XX=X_batch.permute(0,2,3,1)
        X=XX.cpu()
        x=X.numpy().astype(int)
        Y=Y_batch.cpu()
        y=Y.numpy()
        fig=plt.figure()
        plot_3_figs(pred_2,x,y,0)
        # 保存
        plt.savefig(DIR+'lr='+str(LEARNING_RATE)+'\\Train\\time='+str(times + pre)+'.png')
        plt.close('all')
    # 输出中间结果
    if (times + pre)%out_t==0 and (times + pre)!=0:
        ######  方案1：10个out为一组输出【但是内存受限】
#         O=out.unsqueeze(2).cpu() #第一维为样本编号，第二维为输出通道，第三维为训练周期数/100，四五维为像素坐标
#         OUT=torch.cat((OUT,O),2)
#         if times%1000==0 and times!=0:
#             OUT_np=OUT.detach().numpy()
#             np.save('out:'+str(times-1000)+'~'+str(times)+'.npy',OUT_np)
#             OUT=torch.Tensor([])
        ######  方案2：每次out输出
        O=out.cpu().detach().numpy()
        name_temp=DIR+'out\\out_'+str(times + pre)
        np.save(name_temp,O)
        # del out
        # del out_p
    # 切换训练集
    if times%switch_t==0 and times!=0:
        X_train,Y_train=read_data(np.random.randint(DATASET_NUM-1)+1)
        
        
###################################################################### 手动部分        
# 网络参数保存
torch.save(model.state_dict(), os.path.join(DIR,'FCN_Unet_model_5th.pkl'))  # 保存整个网络
# 优化器参数保存
torch.save(optimizer.state_dict(), os.path.join(DIR,"optimizer_state_dict.pkl"))
# torch.save(optimizer,os.path.join(DIR,"optimizer_state_dict.pkl"))

# 手动保存验证用label
if pre == 0:
    np.save(DIR+'lable.npy',y)

# 去掉头就可以吃了。 只有一个头！
TRAIN_LOSS=np.delete(TRAIN_LOSS,0)
TRAIN_ACC=np.delete(TRAIN_ACC,0)
EVAL_LOSS=np.delete(EVAL_LOSS,0)
EVAL_ACC=np.delete(EVAL_ACC,0)
t = np.arange(pre+loss_t, pre+loss_t*(TRAIN_LOSS.size+1), loss_t)

# Loss & Accuracy 数据展示
### 两张图片的scale-Y需要目测更改
print(pre/loss_t+TRAIN_LOSS.size)
# print(TRAIN_ACC)
# print(EVAL_LOSS)
# print(EVAL_ACC)
fig1=plt.figure(1,(8,4))
plt.plot(t,TRAIN_LOSS,t,EVAL_LOSS)
plt.savefig(DIR+'loss_'+str(pre + loss_t)+'_'+str(pre + TRAIN_LOSS.size*loss_t)+'.png')
# plt.xlim((loss_t*(TRAIN_LOSS.size+1))*0.75, loss_t*(TRAIN_LOSS.size+1))
# plt.ylim(0,0.1)
# plt.savefig(DIR+'loss_'+str(pre + loss_t)+'_'+str(pre + TRAIN_LOSS.size*loss_t)+'_局部.png')
fig2=plt.figure(2,(8,4))
plt.plot(t,TRAIN_ACC,t,EVAL_ACC)
plt.savefig(DIR+'accuracy_'+str(pre + loss_t)+'_'+str(pre + TRAIN_LOSS.size*loss_t)+'.png')
# plt.xlim((loss_t*(TRAIN_LOSS.size+1))*0.75, loss_t*(TRAIN_LOSS.size+1))
# plt.ylim(0.75,1)
# plt.savefig(DIR+'accuracy_'+str(pre + loss_t)+'_'+str(pre + TRAIN_LOSS.size*loss_t)+'_局部.png')
plt.show()

# 保存结果  
L_A=np.stack((TRAIN_LOSS,TRAIN_ACC,EVAL_LOSS,EVAL_ACC),1)
print(L_A.shape)
np.save(DIR+'loss_and_acc_'+str(pre + loss_t)+'_'+str(pre + TRAIN_LOSS.size*loss_t)+'.npy',L_A)

