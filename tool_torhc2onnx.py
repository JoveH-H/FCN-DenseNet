import torch
import os
import densenet
from model import Deconv
import argparse

# 文件绝对地址
Absolute_File_Path = os.path.dirname(__file__).replace('\\', '/')

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default='./resources/images/data/test/')  # training dataset
parser.add_argument('--output_dir', default='./resources/images/data/test/')  # training dataset
parser.add_argument('--para_dir', default='./parameters_densenet89/')  # training dataset
parser.add_argument('--b', type=int, default=1)  # batch size
parser.add_argument('--q', default='densenet89')  # save checkpoint parameters
opt = parser.parse_args()
print(opt)

Net_Class_Set = 1

Net_List = ['feature', 'deconv']
Net_Input_List = [torch.rand(1, 3, 320, 240), torch.rand(1, 3, 192, 32, 32)]
# 需要训练的模型类型地址，具体请参考说明文档
Net_Class = Net_List[Net_Class_Set]
Net_Input = Net_Input_List[Net_Class_Set]

# 实例化一个网络对象
if Net_Class_Set == 0:
    model = getattr(densenet, opt.q)(pretrained=True).cpu()
else:
    model = Deconv(opt.q).cpu()

Model_File_Path = Absolute_File_Path + "/model/{}_model.pth".format(Net_Class)
Onnx_File_Path = Absolute_File_Path + "/model/{}_model.onnx".format(Net_Class)
model.load_state_dict(torch.load(Model_File_Path, map_location='cpu'))
model.eval()


def torch2onnx(model, save_path):
    """
    :param model:pkl
    :param save_path:onnx
    :return:onnx
    """
    model.eval()
    data = Net_Input
    input_names = ["{}_input".format(Net_Class)]
    output_names = ["{}_out".format(Net_Class)]
    torch.onnx._export(model, data, save_path, export_params=True, opset_version=11, input_names=input_names, output_names=output_names)
    input("torch2onnx finish. 任意键退出...")


if __name__ == '__main__':
    torch2onnx(model, Onnx_File_Path)
