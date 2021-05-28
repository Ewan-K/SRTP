import numpy as np
import matplotlib.pyplot as plt


Batch = 120


def line(startx, starty, endx, endy, path_image):
    # startx 和 starty 是线条出发点的横纵坐标
    # endx 和 endy 是线条到达点的横纵坐标
    # path_image 是路径图像
    n1 = abs(endx - startx)
    n2 = abs(endy - starty)
    n = max(n1, n2)  # 路径像素点总数
    coincide = 0  # 重合次数
    if n == n1:
        if startx >= endx:
            for x in range(endx, startx):
                y = int((x - startx) * (endy - starty) / (endx - startx)) + starty
                # 注：额外判断线条上下一个像素点，可以增加准确率
                if path_image[x, y] == 1 or path_image[x, y + 1] == 1 or path_image[x, y - 1] == 1:
                    coincide = coincide + 1
        else:
            for x in range(startx, endx):
                y = int((x - startx) * (endy - starty) / (endx - startx)) + starty
                if path_image[x, y] == 1 or path_image[x, y + 1] == 1 or path_image[x, y - 1] == 1:
                    coincide = coincide + 1
    if n == n2 and n != n1:
        if starty >= endy:
            for y in range(endy, starty):
                x = int((y - starty) * (endx - startx) / (endy - starty)) + startx
                if path_image[x, y] == 1 or path_image[x - 1, y] == 1 or path_image[x + 1, y] == 1:
                    coincide = coincide + 1
        else:
            for y in range(starty, endy):
                x = int((y - starty) * (endx - startx) / (endy - starty)) + startx
                if path_image[x, y] == 1 or path_image[x - 1, y] == 1 or path_image[x + 1, y] == 1:
                    coincide = coincide + 1

    return coincide / n  # 返回 重合次数/路径像素点总数


def belief_value(city_site, path_image):
    # 输入城市坐标 与 路径图像
    A = np.zeros([NUM_CITY, NUM_CITY])  # 邻接矩阵
    # 对于A逐行逐列生成连线概率值,计算上三角矩阵
    for row in range(NUM_CITY - 1):
        for col in range(row + 1, NUM_CITY):
            A[row, col] = line(city_site[row, 0], city_site[row, 1], city_site[col, 0], city_site[col, 1], path_image)
    # 下三角矩阵采用上三角矩阵转置生成
    for row in range(NUM_CITY):
        for col in range(row, NUM_CITY):
            A[col, row] = A[row, col]
    return A


def distance(p1, p2):
    p3 = p2 - p1
    # print(p3)
    dis = np.linalg.norm(p2 - p1)
    return dis


total_diff = 0.0
for k in range(Batch):
    path_image = np.load("TSP_Unet_out_1/lr=0.0003_1/out/" + str(k) + ".npy")
    city_site = np.load("partial_npy/" + str(k) + ".npy")
    NUM_CITY = len(city_site)
    city_site = city_site * 2
    # 结合实际坐标来看，神经网络的输出坐标变小了，可能是png转npy代码的问题
    for i in range(len(city_site)):  # 坐标需要换回先y后x
        city_site[i][0], city_site[i][1] = city_site[i][1], city_site[i][0]
    # print(city_site)
    A = belief_value(city_site, path_image)
    # 生成城市搜索次序表
    City_order = np.arange(NUM_CITY)
    # 以A中第0行出发，找出下一个城市
    Next_City = 0
    Total = 0.0
    print("出 发 城 市:", city_site[Next_City])
    for search_city in range(1, NUM_CITY):
        Curr_City = Next_City
        # 将已经经过的城市从搜索次序表中剔除
        City_order = City_order[np.where(City_order != Next_City)]
        # 在未经过的城市之中搜索
        biggest_prob = 0
        for i in range(NUM_CITY):
            if i in City_order and A[Next_City, i] > biggest_prob:
                biggest_prob = A[Next_City, i]
                Next_City = i
        Total += distance(city_site[Curr_City], city_site[Next_City])
        print("第", search_city, "个城市:", end=" ")
        print(city_site[Next_City])
    Total += distance(city_site[0], city_site[Next_City])  # 回到出发城市

    # 计算误差
    path = "data/" + str(k + 1) + ".txt"
    diff = 0.0
    with open(path, "r") as f:
        for l in range(13):
            data = f.readline()  # read the redundant lines
        standard = f.readline()
        target_data = standard.split()
        diff = abs(float(target_data[2]) - Total) / float(target_data[2])
        # print(diff)
        total_diff += diff
total_diff /= Batch
print(total_diff)
