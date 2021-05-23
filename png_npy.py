import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

batch_size = 20  # 测试集中MTSP样例个数
SIZE = 6  # 一个文件内存储的图片个数（MTSP样例个数*6）
num = 6  # 一个样例中旅行商个数

# 输入路径
Path = ".\\partial_pic\\"
# 输出路径
fname = '.\\KMeans_out\\'

img = np.zeros((SIZE, 448, 448, 3), dtype=np.int)
mtr = np.zeros((SIZE, 448, 448), dtype=np.int)

f_save = fname + 'KMeans_'
j = 0  # 用于标记是否填满一个npy文件
l = 0  # 用于标记npy文件个数
for i in range(batch_size):
    for k in range(num):
        path_temp = Path + str(i + 1) + "_" + str(k + 1) + ".png"
        img0 = Image.open(path_temp)
        img0 = img0.convert("RGB")  # png文件是RGBA四通道，需要转换成三通道
        img[j] = np.array(img0)
        for p in range(448):  # input图片像素行数
            for q in range(448):  # input图片像素列数
                if (all(img[j][p][q] == (255, 255, 255))):
                    mtr[j][p][q] = 0  # 背景代号为0
                elif (all(img[j][p][q] == (0, 0, 255))):
                    mtr[j][p][q] = 1  # 途径路线代号为1
                else:
                    mtr[j][p][q] = 2  # 途径地代号为2
        j = j + 1
        if j == SIZE:
            fis = f_save + str(l + 1) + '.npy'
            np.save(fis, mtr)  # 保存input
            j = 0
            l = l + 1
