import cv2 as cv
import numpy as np
import os


# 图像的裁剪和合成  https://blog.csdn.net/qq_40608730/article/details/109190493
pic_path = 'D:\\wgr\\hr_1\\roicutnew.tif' # 分割的图片的位置
pic_target = 'D:\\wgr\\hr_1\\data\\cuttifmask\\' # 分割后的图片保存的文件夹
if not os.path.exists(pic_target):  #判断是否存在文件夹如果不存在则创建为文件夹
    os.makedirs(pic_target)
#要分割后的尺寸
cut_width = 1200
cut_length = 1200
# 读取要分割的图片，以及其尺寸等数据
picture = cv.imread(pic_path)
(width, length, depth) = picture.shape
# 预处理生成0矩阵
pic = np.zeros((cut_width, cut_length, depth))
# 计算可以划分的横纵的个数
num_width = int(width / cut_width)
num_length = int(length / cut_length)
# for循环迭代生成
for i in range(0, num_width):
    for j in range(0, num_length):
        pic = picture[i*cut_width : (i+1)*cut_width, j*cut_length : (j+1)*cut_length, :]      
        result_path = pic_target + 'mask'+'{}+{}.png'.format(i+1, j+1)
        cv.imwrite(result_path, pic)
 
print("done!!!")