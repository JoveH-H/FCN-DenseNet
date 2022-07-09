import os
from shutil import copyfile, move
import random

absolute_path = os.path.abspath(os.path.dirname(__file__)).replace('\\', '/')
dataset_dir_path = absolute_path + '/resources/images/dataset'
data_dir_path = absolute_path + '/resources/images/data'


# config
data_type = 1
movement_type = 0
img_id_num = 6
test_ratio = 0.1

movement_type_list = ['copy', 'move']
movement_type = movement_type_list[movement_type]

input("任意键开始")

FileNameList = os.listdir(dataset_dir_path)
FileNameList.sort()  # 排序
random.shuffle(FileNameList)
len_img = len(FileNameList)

now_id = 0

for i in range(len_img):
    #  判断当前文件是否为_img文件
    if os.path.splitext(FileNameList[i])[0][-4:] == "_img":
        now_id += 1
        if now_id > len_img / 2 * test_ratio:
            data_type_path = data_dir_path + '/train'
        else:
            data_type_path = data_dir_path + '/test'

        if movement_type == 'copy':
            copyfile(dataset_dir_path + '/' + FileNameList[i], data_type_path + '/img/' + FileNameList[i][0:img_id_num] + '.png')
            copyfile(dataset_dir_path + '/' + os.path.splitext(FileNameList[i])[0][:-4] + '_label.png', data_type_path + '/mask/' + FileNameList[i][0:img_id_num] + '.png')
        else:
            move(dataset_dir_path + '/' + FileNameList[i], data_type_path + '/img/' + FileNameList[i][0:img_id_num] + '.png')
            move(dataset_dir_path + '/' + os.path.splitext(FileNameList[i])[0][:-4] + '_label.png', data_type_path + '/mask/' + FileNameList[i][0:img_id_num] + '.png')

input("任意键退出")
