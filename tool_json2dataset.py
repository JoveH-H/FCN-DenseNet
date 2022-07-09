import base64
import json
import os
import os.path as osp

import imgviz
import PIL.Image

from labelme import utils


absolute_path = os.path.abspath(os.path.dirname(__file__)).replace('\\', '/')
images_dir_path = absolute_path + '/resources/images/data_json'
dataset_dir_path = absolute_path + '/resources/images/dataset'


def main(json_file_dir=images_dir_path, out_dir=dataset_dir_path):

    input("任意键开始")

    if not osp.exists(out_dir):
        os.mkdir(out_dir)

    FileNameList = os.listdir(json_file_dir)

    for i in range(len(FileNameList)):
        #  判断当前文件是否为json文件
        if os.path.splitext(FileNameList[i])[1] == ".json":
            json_file = json_file_dir + '/' + FileNameList[i]

            data = json.load(open(json_file))
            imageData = data.get("imageData")

            if not imageData:
                imagePath = os.path.join(os.path.dirname(json_file), data["imagePath"])
                with open(imagePath, "rb") as f:
                    imageData = f.read()
                    imageData = base64.b64encode(imageData).decode("utf-8")
            img = utils.img_b64_to_arr(imageData)

            label_name_to_value = {"_background_": 0}
            for shape in sorted(data["shapes"], key=lambda x: x["label"]):
                label_name = shape["label"]
                if label_name in label_name_to_value:
                    label_value = label_name_to_value[label_name]
                else:
                    label_value = len(label_name_to_value)
                    label_name_to_value[label_name] = label_value

            lbl = utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)

            label_names = [None] * (max(label_name_to_value.values()) + 1)
            for name, value in label_name_to_value.items():
                label_names[value] = name

            lbl_viz = imgviz.label2rgb(
                label=lbl, img=imgviz.asgray(img), label_names=label_names, loc="rb"
            )

            PIL.Image.fromarray(img).save(osp.join(out_dir, "{}_img.png".format(os.path.splitext(FileNameList[i])[0])))
            utils.lblsave(osp.join(out_dir, "{}_label.png".format(os.path.splitext(FileNameList[i])[0])), lbl)

            print("Finish to: {}".format(json_file))

    input("任意键退出")


if __name__ == "__main__":
    main()
