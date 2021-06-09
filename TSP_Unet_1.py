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
# from torch import optim
# from torch import autograd
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")
# from memory_profiler import profile


# 定义需要的全局变量
PRE=0
SIZE=6
DATASET_NUM = 20
PATH='.\\0数据集\\TSP\\Kmeans_out\\'   #数据保存位置目录名称
F_IN = PATH+'KMeans_'
# F_OUT = PATH+'MTSP_N30M6P10_Out'
DIR='.\\Kmeans_app\\'
DIR1=".\\tsp_2\\Unet_model_tsp.pkl"
DIMENSION_STATE=448 # 图像的大小
BATCH_SIZE=6 #读取数据的批大小
CLASS_NUM=2 #输出像素类型数量
LEARNING_RATE=3E-4 # 学习率(e-3~e-5之间，3倍为一档)
# momentum = 0.5 # 优化算法的超参数（0.5~0.9之间）

# 目录准备
# if not os.path.exists(DIR+'lr='+str(LEARNING_RATE)+'\\Comparison'):
#     os.makedirs(DIR+'lr='+str(LEARNING_RATE)+'\\Comparison')
if not os.path.exists(DIR+'lr='+str(LEARNING_RATE)+'\\out_Image'):
    os.makedirs(DIR+'lr='+str(LEARNING_RATE)+'\\out_Image')
if not os.path.exists(DIR+'lr='+str(LEARNING_RATE)+'\\out'):
    os.makedirs(DIR+'lr='+str(LEARNING_RATE)+'\\out')

# 函数：从不同的文件中读取数据
def read_data(signal):
    X_dir=F_IN+str(signal)+".npy"
    # Y_dir=F_OUT+str(signal+1)+".npy"
    X_train=np.load(X_dir)
    # Y_train=np.load(Y_dir)
    return X_train  #,Y_train

# 函数：从读取的数据中定向获取部分样本进行训练
def direct_batch(X_test,org,size):
    if org+size>SIZE:
        org=SIZE-size
    X_batch = X_test[org:(org+size)]
    X_batch=torch.from_numpy(X_batch)# 转换成tensor
    X_batch=X_batch.type(torch.FloatTensor) # 将tensor统一为float64格式  #数据类型对训练速度没有太大影响
    return X_batch

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
        ##########################
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


# # %xdel model

# 实例化网络
model=Unet(1,CLASS_NUM)
model.to(device)

# # 定义损失函数
# # criterion = nn.CrossEntropyLoss()
# # 带权重
# p_w=np.array([1,5])
# P_W=torch.from_numpy(p_w).to(device)
# criterion = nn.BCEWithLogitsLoss(pos_weight=P_W)
# # # 不带权重
# # criterion = nn.BCEWithLogitsLoss()

# 网络参数提取
state=torch.load(DIR1)
model.load_state_dict(state)

# @profile
# def plot_3_figs(pred,x,y,n):
#     plt.subplot(131)
#     plt.imshow(x[n,:,:,:],cmap=plt.cm.gray)
#     plt.title('input')
#     plt.axis('off')  #去掉坐标轴
#     plt.subplot(132)
#     plt.imshow(pred[n,:,:],cmap=plt.cm.gray)
#     plt.title('output')
#     plt.axis('off')  #去掉坐标轴
#     plt.subplot(133)
#     plt.imshow(y[n,:,:],cmap=plt.cm.gray)
#     plt.title('label')
#     plt.axis('off')  #去掉坐标轴
    
  
pre=0  # 只能是5的倍数

# 从文件中提取数据
X_test=read_data(1)

for times in range(pre,120):
    if times%BATCH_SIZE!=0:
        continue
    else:
        # 切换训练集
        if times%SIZE==0:
            X_test=read_data(times//SIZE+1)
    # *********************************进入测试模式*************************************
        model.eval()
        # 读取数据
        X_test_batch=direct_batch(X_test,times%SIZE,BATCH_SIZE) 
        X_test_batch = X_test_batch.to(device)
        # Y_test_batch = Y_test_batch.to(device)
        X_test_batch = X_test_batch.unsqueeze(1) 
        # oh_Y_test_batch=torch.nn.functional.one_hot(Y_test_batch.long(),CLASS_NUM)
        # oh_Y_test_batch=oh_Y_test_batch.float()
        # 前向传播
        out = model(X_test_batch)
        out_p=out.permute(0,2,3,1)
        # loss = criterion(out_p,oh_Y_test_batch)
        # 记录误差
        # eval_loss += loss.item()/BATCH_SIZE
        # 记录测试的准确率
        _,eval_pred = out.max(1)
        # *********************************  记录 *************************************
        print("当前处理：第",times+1,"~",times+6,"个案例")
        print("------------------------------------------------------")
        # ***************************** 【pred_2为所求的np.array】*****************
        pred_1=eval_pred.cpu()
        pred_2=pred_1.numpy()
        XX=X_test_batch.permute(0,2,3,1)
        X=XX.cpu()
        x=X.numpy().astype(int)
        # Y=Y_test_batch.cpu()
        # y=Y.numpy()
        for i in range(BATCH_SIZE): ###############输出5个验证集数据的训练过程
            # fig=plt.figure()
            # plot_3_figs(pred_2,x,y,i)
            # plt.savefig(DIR+'lr='+str(LEARNING_RATE)+'\\Comparison\\'+str(times + i)+'.png')
            # plt.close('all')
            
            fig=plt.figure()
            plt.imshow(pred_2[i,:,:],cmap=plt.cm.gray)
            plt.axis('off')  #去掉坐标轴
            plt.savefig(DIR+'lr='+str(LEARNING_RATE)+'\\out_Image\\'+str(times + i)+'.png')
            plt.close('all')
            np.save(DIR+'lr='+str(LEARNING_RATE)+'\\out\\'+str(times + i)+'.npy',pred_2[i,:,:])
            
            
###################################################################### 手动部分        

# # 手动保存验证用label
# if pre == 0:
#     np.save(DIR+'lable.npy',y)

# # 去掉头就可以吃了。 只有一个头！
# TRAIN_LOSS=np.delete(TRAIN_LOSS,0)
# TRAIN_ACC=np.delete(TRAIN_ACC,0)
# EVAL_LOSS=np.delete(EVAL_LOSS,0)
# EVAL_ACC=np.delete(EVAL_ACC,0)
# t = np.arange(pre+loss_t, pre+loss_t*(TRAIN_LOSS.size+1), loss_t)

# # Loss & Accuracy 数据展示
# ### 两张图片的scale-Y需要目测更改
# print(pre/loss_t+TRAIN_LOSS.size)
# # print(TRAIN_ACC)
# # print(EVAL_LOSS)
# # print(EVAL_ACC)
# fig1=plt.figure(1,(8,4))
# plt.plot(t,TRAIN_LOSS,t,EVAL_LOSS)
# plt.savefig(DIR+'loss_'+str(pre + loss_t)+'_'+str(pre + TRAIN_LOSS.size*loss_t)+'.png')
# # plt.xlim((loss_t*(TRAIN_LOSS.size+1))*0.75, loss_t*(TRAIN_LOSS.size+1))
# # plt.ylim(0,0.1)
# # plt.savefig(DIR+'loss_'+str(pre + loss_t)+'_'+str(pre + TRAIN_LOSS.size*loss_t)+'_局部.png')
# fig2=plt.figure(2,(8,4))
# plt.plot(t,TRAIN_ACC,t,EVAL_ACC)
# plt.savefig(DIR+'accuracy_'+str(pre + loss_t)+'_'+str(pre + TRAIN_LOSS.size*loss_t)+'.png')
# # plt.xlim((loss_t*(TRAIN_LOSS.size+1))*0.75, loss_t*(TRAIN_LOSS.size+1))
# # plt.ylim(0.75,1)
# # plt.savefig(DIR+'accuracy_'+str(pre + loss_t)+'_'+str(pre + TRAIN_LOSS.size*loss_t)+'_局部.png')
# plt.show()

# # 保存结果  
# L_A=np.stack((TRAIN_LOSS,TRAIN_ACC,EVAL_LOSS,EVAL_ACC),1)
# print(L_A.shape)
# np.save(DIR+'loss_and_acc_'+str(pre + loss_t)+'_'+str(pre + TRAIN_LOSS.size*loss_t)+'.npy',L_A)

