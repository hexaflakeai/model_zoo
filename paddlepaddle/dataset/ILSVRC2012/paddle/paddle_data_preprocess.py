import os
import cv2
import tarfile
import argparse
import numpy as np
import tvm.contrib.hexaflake.vision.ocr as hxocr
import tvm.contrib.hexaflake.tools.utils as hxutils
import tvm.contrib.hexaflake.tools.common as hxcomm


class ClassificationDataProcess():
    def __init__(self, gt_path, save_folder):
        with open(gt_path) as f:
            labels = [i.strip() for i in f.readlines()]
        self.targets = {}
        for i in labels:
            img_name, label = i.split()
            self.targets[img_name] = label

        self.transform_list = [
                {"ResizeImage": {"resize_short": 256}},
                {"CropImage": {"size": 224}},
                {"NormalizeImage": {"scale": 0.00392157, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "order": ""}},
                {"ToCHWImage": None},
                {"KeepKeys": {"keep_keys": ["image"]}},
            ]
        self.func_load_input = hxocr.load_input_tvm_dbnet

        os.makedirs(os.path.join(save_folder, "info"), exist_ok=True)
        os.makedirs(os.path.join(save_folder, "input"), exist_ok=True)
        self.save_folder = save_folder

    def preprocess(self, img_path, name):
        trans_img = hxutils.deal_with_variable_args(
            self.func_load_input, img_path, self.transform_list
        )
        data_out = trans_img[0].tobytes()
        label_out = {"label": [self.targets[name]]}
        return data_out, label_out

    def export_dataset_tar(self):
        os.system(f"tar -zcvf {self.save_folder.split('/')[-1]}.tar.gz {self.save_folder}")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="HX arg parser", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--image_folder", "-img", help="imagenet2012 val image", type=str, default="./ILSVRC2012_img_val")
    parser.add_argument("--label_txt", "-label", help="imagenet2012 val label", type=str, default="")
    args = parser.parse_args()

    image_folder = args.image_folder
    gt_path = args.label_txt
    save_folder = "./paddle_classification_datasets"
    img_names = os.listdir(image_folder.split("/")[-1])
    ils2012_process = ClassificationDataProcess(gt_path, save_folder)
    for idx, name in enumerate(img_names):
        print(f"idx: {idx+1}/{img_names.__len__()}", "\r", end="", flush=True)
        img_path = os.path.join(image_folder.split("/")[-1], name)
        data, label = ils2012_process.preprocess(img_path, name)
        with open(f"{save_folder}/input/input-{idx}.bin", "wb") as f:
            f.write(data)
        np.save(f"{save_folder}/info/info-{idx}.npy", label)
    ils2012_process.export_dataset_tar()

