import os
import numpy as np
import imgviz
from PIL import Image

absolute_path = os.path.abspath(os.path.dirname(__file__)).replace('\\', '/')
data_dir_path = absolute_path + '/resources/images/annotations/trimaps'

ImgFileNameList = os.listdir(data_dir_path)


def get_gray_cls(van_lbl, array_lbl):
    cls = [2, 3]  # 用来存储灰度图像中每种类别所对应的像素，默认背景色为0
    for x in range(van_lbl.size[0]):
        for y in range(van_lbl.size[1]):
            if array_lbl[y, x] not in cls:
                cls.append(array_lbl[x, y])
    return cls


def get_P_cls(cls_gray):
    cls_P = []  # 将灰度图像中的每类像素用0~N表示
    for i in range(len(cls_gray)):
        cls_P.append(i)
    return cls_P


def array_gray_to_P(cls_gray, cls_P, array):
    for i in range(len(cls_gray)):
        array[array == cls_gray[i]] = cls_P[i]
    return array


if __name__ == '__main__':
    van_file = data_dir_path + '/Abyssinian_1.png'  # 必须是一张包含所有类别的图像,称之为先锋图像
    van_lbl = Image.open(van_file).convert('L')  # 将先锋图像转换为灰度图像

    array_lbl = np.array(van_lbl)  # 获得灰度图像的numpy矩阵

    cls_gray = get_gray_cls(van_lbl, array_lbl)  # 获取灰度图像中每种类别所对应的像素值
    cls_P = get_P_cls(cls_gray)  # 将灰度图像中的每种类别所对应的像素值映射为0~N

    # 遍历每一张原始图像
    len_img = len(ImgFileNameList)
    for i in range(len_img):
        orig_lbl = Image.open(data_dir_path + '/' + ImgFileNameList[i]).convert('L')  # 将图像转换为灰度图像
        array_gray = np.array(orig_lbl)  # 获得灰度图像的numpy矩阵
        array_P = array_gray_to_P(cls_gray, cls_P, array_gray)  # 将灰度图像的numpy矩阵值映射为0~N
        label = Image.fromarray(array_P.astype(np.uint8), mode='P')  # 转换为PIL的P模式
        # 转换成VOC格式的P模式图像
        colormap = imgviz.label_colormap()
        label.putpalette(colormap.flatten())
        label.save(data_dir_path + '/' + ImgFileNameList[i])
