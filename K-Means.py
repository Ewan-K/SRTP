import math
import random
import numpy as np
import matplotlib.pyplot as plt

NUM = 20  # 待处理的样例个数


def getEuclidean(point1, point2):
    dimension = len(point1)
    dist = 0.0
    for i in range(dimension):
        dist += (point1[i] - point2[i])**2
    return math.sqrt(dist)


def kMeans(dataset, k, iteration):
    # 初始化簇心向量
    index = random.sample(list(range(len(dataset))), k)
    vectors = []
    for i in index:
        vectors.append(dataset[i])
    # 初始化所有城市所属标签
    labels = []
    for i in range(len(dataset)):
        labels.append(-1)
    # 根据迭代次数重复聚类过程
    while (iteration > 0):
        # 初始化k个簇并将中心城市加入每个簇中
        C = []
        for i in range(k):
            C.append([])
            C[i].append(dataset[0])
        # 更新每个城市所属簇
        for labelIndex, item in enumerate(dataset):
            if labelIndex:  # 跳过簇心（已经初始化了）
                classIndex = -1
                minDist = 1e6
                for i, point in enumerate(vectors):
                    dist = getEuclidean(item, point)
                    if (dist < minDist):
                        classIndex = i
                        minDist = dist
                C[classIndex].append(item)
                labels[labelIndex] = classIndex
        # 更新簇心向量
        for i, cluster in enumerate(C):
            clusterHeart = []
            dimension = len(dataset[0])
            # clusterHeart与城市坐标维数相同
            for j in range(dimension):
                clusterHeart.append(0)
            for item in cluster:
                for j, coordinate in enumerate(item):
                    clusterHeart[j] += coordinate / len(cluster)
            vectors[i] = clusterHeart
        iteration -= 1
    # 约束簇中元素个数在lower和upper之间（包括中心城市）
    lower = 4
    upper = 7

    # 忍痛割爱
    for i in range(k):
        while (len(C[i]) > upper):
            target = -1
            minDist = 1e6
            # 寻找距离本簇簇心最近的另一个簇心
            for j in range(k):
                if j != i:  # 排除本簇
                    dist = getEuclidean(vectors[i], vectors[j])
                    if dist < minDist and len(C[j]) < upper:
                        target = j
                        minDist = dist
                else:
                    continue
            index = -1
            minDist = 1e6
            # 寻找本簇中距离该簇簇心最近的点
            for j, item in enumerate(C[i]):
                if j != 0:  # 排除depot
                    dist = getEuclidean(item, vectors[target])
                    if dist < minDist:
                        index = j
                        minDist = dist
                else:
                    continue
            # 从本簇归入目标簇
            item = C[i].pop(index)
            C[target].append(item)
            labels[dataset.index(item)] = target  # 修改labels

    # 夺人所爱
    for i in range(k):
        while (len(C[i]) < lower):
            target = -1
            minDist = 1e6
            # 寻找簇外距离本簇簇心最近的点（也可以仿照忍痛割爱，先找距离本簇簇心最近的另一个簇心，再找该簇中距离本簇最近的点）
            for j in range(len(dataset)):
                if labels[j] != i and j != 0:  # 排除本簇和depot
                    dist = getEuclidean(dataset[j], vectors[i])
                    if dist < minDist and len(C[labels[j]]) > lower:
                        target = j
                        minDist = dist
                else:
                    continue
            # 从原所属簇归入本簇
            C[labels[target]].remove(dataset[target])
            C[i].append(dataset[target])
            labels[target] = i  # 修改labels

    return C


# 获取数据集并调用K-Means处理
pathIn = "./data/"
pathOut1 = "./full/"
pathOut2 = "./partial_npy/"  # 每簇的城市坐标（未转化到448）
pathOut3 = "./partial_pic/"  # 每簇的城市坐标画成的图（转化到448）
idx = 0  # 标记局部城市坐标文件
for k in range(NUM):
    fname = pathIn + str(k + 1) + '.txt'
    with open(fname, "r") as f:
        for l in range(10):
            data = f.readline()  # read the redundant lines
        y = f.readline()  # y 先读y再读x来转化x, y的颠倒关系
        data_y = y.split()
        x = f.readline()  # x
        data_x = x.split()
        # 字符串转整数
        for l, num in enumerate(data_x):
            data_x[l] = int(num)
        for l, num in enumerate(data_y):
            data_y[l] = int(num)
        dataset = list(zip(data_x, data_y))
        # 数据处理
        C = kMeans(dataset, 6, 2000)

        # 画整体图
        colValue = ['r', 'y', 'g', 'b', 'c', 'k', 'm']
        for i in range(len(C)):
            coo_X = []  # x坐标列表
            coo_Y = []  # y坐标列表
            # 得到一簇所有点的坐标
            for j in range(len(C[i])):
                coo_X.append(C[i][j][0] * 2)  # 转化到448*448
                coo_Y.append(C[i][j][1] * 2)
            plt.scatter(coo_X,
                        coo_Y,
                        marker='x',
                        color=colValue[i % len(colValue)],
                        label=i)
        plt.xlim((0, 448))
        plt.ylim((448, 0))
        plt.axis('off')
        plt.savefig(pathOut1 + str(k + 1) + ".png")

        plt.cla()

        # 存储每个簇的城市坐标
        for i in range(6):
            np.save(pathOut2 + str(idx) + ".npy", C[i])
            idx = idx + 1

        # 画局部图
        for i in range(len(C)):
            coo_X = []  # x坐标列表
            coo_Y = []  # y坐标列表
            # 得到一簇所有点的坐标
            for j in range(len(C[i])):
                coo_X.append(C[i][j][0] * 2)  # 转化到448*448
                coo_Y.append(C[i][j][1] * 2)
            plt.figure(figsize=(4.48, 4.48), dpi=100)  # 规定像素级尺寸
            for j in range(len(C[i])):
                for l in range(len(C[i])):
                    x = []
                    y = []
                    x.append(coo_X[j])
                    x.append(coo_X[l])
                    y.append(coo_Y[j])
                    y.append(coo_Y[l])
                    plt.plot(x, y, color='b')
            plt.scatter(coo_X, coo_Y, marker='s', color='k', label=i)
            plt.xlim((0, 448))
            plt.ylim((448, 0))
            plt.axis('off')
            plt.savefig(pathOut3 + str(k + 1) + "_" + str(i + 1) + ".png")
            plt.cla()
            plt.close()  # 以防plt画布重叠
