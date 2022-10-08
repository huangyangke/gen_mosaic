import cv2
import numpy as np
from  matplotlib import pyplot as plt
import random
from joblib import delayed
from joblib import Parallel
import os
import time

###########################输入图像roi区域和想要的马赛克块大小，在原图上进行马赛克化###########################
def normal_mosaic(selected_img, mosaic_width):
    height, width, _= selected_img.shape
    #遍历左上角点
    for m in range(0, height, mosaic_width):
        for n in range(0, width, mosaic_width):
        #如果该像素点为卷积核左上角的点 则该卷积核内的其他值均为该点的rgb值
            selected_img[m:m+mosaic_width,n:n+mosaic_width] = selected_img[m,n]
    #plt.imshow(selected_img[:,:,::-1])#plt是rgb读取 这里反向一下  

img = cv2.imread("D:/wangyi/data/mosaic/internet_image/no_mosaic/1.jpg")
normal_mosaic(img, 10)
plt.imshow(img[:,:,::-1])

###########################高斯马赛克###########################
def gaussian_mosaic(selected_img):
    height, width, _= selected_img.shape
    selected_img[:,:,0] = np.random.normal(size = selected_img.shape[:2])
    selected_img[:,:,1] = np.random.normal(size = selected_img.shape[:2])
    selected_img[:,:,2] = np.random.normal(size = selected_img.shape[:2])
img = cv2.imread("D:/wangyi/data/mosaic/internet_image/no_mosaic/1.jpg")
gaussian_mosaic(img)
plt.imshow(img[:,:,::-1])

###########################毛玻璃马赛克###########################
#使用当前领域内的某个像素点来替换当前像素点
def ground_glass(selected_img):
    height, width, _= selected_img.shape
    dst = selected_img.copy()
    mosaic_width = 8 #在8邻域内
    for m in range(height - mosaic_width):
        for n in range(width - mosaic_width):
            index = random.randint(0,mosaic_width)
            b, g, r = selected_img[m + index, n + index]
            dst[m, n] = b, g, r
    selected_img[:,:,:] = dst
    
img = cv2.imread("D:/wangyi/data/mosaic/internet_image/no_mosaic/1.jpg")
ground_glass(img)
plt.imshow(img[:,:,::-1])

###########################竖条纹马赛克###########################
def Vertical_stripes_mosaic(selected_img):
    height, width, _= selected_img.shape
    stripes_width = random.randint(5, 20)#竖条纹宽度范围暂定3-10
    num = 0
    for n in range(0, width, stripes_width):
        if num % 2 == 0:
            r, g, b = 0, 0, 0
        else:
            r, g, b = 255, 255, 255
        selected_img[:,n:n+stripes_width] = r, g, b
        num += 1

img = cv2.imread("D:/wangyi/data/mosaic/internet_image/no_mosaic/1.jpg")
Vertical_stripes_mosaic(img)
plt.imshow(img[:,:,::-1])

###########################黑白相间马赛克###########################
def black_white_mosaic(selected_img, mosaic_width):
    height, width, _= selected_img.shape
    num = 0
    #遍历左上角点
    for m in range(0, height, mosaic_width):
        for n in range(0, width, mosaic_width):
        #如果该像素点为卷积核左上角的点 则该卷积核内的其他值均为该点的rgb值
            if num % 2 == 0:
                selected_img[m:m+mosaic_width,n:n+mosaic_width] = 0, 0, 0
            else:
                selected_img[m:m+mosaic_width,n:n+mosaic_width] = 255, 255, 255     
            num += 1
        num += 1

img = cv2.imread("D:/wangyi/data/mosaic/internet_image/no_mosaic/1.jpg")
black_white_mosaic(img, 10)
plt.imshow(img[:,:,::-1])

###########################单一图片执行生成mosaic###########################
# 输入图像路径 生成马赛克 并且保存
def mosaic_create(image_path, train_or_val):
    # 在图像上随机大小的矩形区域生成马赛克
    img = cv2.imread(image_path)  # bgr读取
    # print(img)
    height, width, deep = img.shape
    mosaic_width = random.randint(2, 50)  # 随机生成马赛克块大小 这里暂定为2到50
    # mosaic_width = 10

    rectangle_width = random.randint(int(width / 25), int(width / 2))  # 随机生成马赛克区域大小
    rectangle_height = random.randint(int(height / 25), int(height / 2))

    rectangle_y = random.randint(0, height - rectangle_height)
    rectangle_x = random.randint(0, width - rectangle_width)

    # print(rectangle_y,rectangle_x,rectangle_y+rectangle_height,rectangle_width+rectangle_x)#左上，右下
    # print(height,width)

    probability_value = random.randint(0, 100)

    #这里设置50%的概率生成常规马赛克，30%的概率生成毛玻璃马赛克，10%的概率生成竖条纹马赛克，10%的概率生成黑白相间马赛克
    if probability_value > 50:
        normal_mosaic(img[rectangle_y: rectangle_y + rectangle_height, rectangle_x:rectangle_width + rectangle_x],mosaic_width)
    elif probability_value > 20:
        ground_glass(img[rectangle_y: rectangle_y + rectangle_height, rectangle_x:rectangle_width + rectangle_x])
    elif probability_value > 10:
        Vertical_stripes_mosaic(img[rectangle_y: rectangle_y + rectangle_height, rectangle_x:rectangle_width + rectangle_x])
    else:
        black_white_mosaic(img[rectangle_y: rectangle_y + rectangle_height, rectangle_x:rectangle_width + rectangle_x],mosaic_width)
    label = img.copy()
    label[:, :, :] = 0
    label[rectangle_y: rectangle_y + rectangle_height, rectangle_x:rectangle_width + rectangle_x] = np.array(
        [128, 128, 128])  # 标签绘制

    # 生成的马赛克图片保存
    name = image_path.split('/')[-1]
    if train_or_val == 'train':
        save_image_path = os.path.join(newtrain_image_path, name)
    else:
        save_image_path = os.path.join(newval_image_path, name)

    cv2.imwrite(save_image_path, img)

    # 生成的label图片保存
    name = name.replace('jpg', 'png')  # 保存的格式换一下
    if train_or_val == 'train':
        save_label_path = os.path.join(train_label_path, name)
    else:
        save_label_path = os.path.join(val_label_path, name)

    cv2.imwrite(save_label_path, label)
