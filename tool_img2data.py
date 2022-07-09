import os
from shutil import copyfile
import random

absolute_path = os.path.abspath(os.path.dirname(__file__)).replace('\\', '/')
img_dir_path = absolute_path + '/resources/images/images'
lab_dir_path = absolute_path + '/resources/images/annotations/trimaps'
data_dir_path = absolute_path + '/resources/images/data'


# config
data_type = 1
movement_type = 0
img_id_num = 6
test_ratio = 0.03

input("任意键开始")


ImgFileNameList = os.listdir(img_dir_path)
LabFileNameList = os.listdir(lab_dir_path)
ImgFileNameList.sort()  # 排序
random.shuffle(ImgFileNameList)

len_img = len(ImgFileNameList)
for i in range(len_img):
    if i + 1 > len_img * test_ratio:
        copyfile(img_dir_path + '/' + ImgFileNameList[i], data_dir_path + '/train/img/' + ImgFileNameList[i])
        copyfile(lab_dir_path + '/' + ImgFileNameList[i], data_dir_path + '/train/mask/' + ImgFileNameList[i])
    else:
        copyfile(img_dir_path + '/' + ImgFileNameList[i], data_dir_path + '/test/img/' + ImgFileNameList[i])
        copyfile(lab_dir_path + '/' + ImgFileNameList[i], data_dir_path + '/test/mask/' + ImgFileNameList[i])

input("任意键退出")
