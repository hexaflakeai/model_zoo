import os
import cv2
import json
import shutil
import paddle
import argparse
import numpy as np
from tqdm import tqdm
import subprocess as sp

import tvm.contrib.hexaflake.vision.ocr as hxocr
import tvm.contrib.hexaflake.tools.utils as hxutils
from tvm.contrib.paddle import ICDAR_utils, DetMetric


class OCRDetectionValid:
    def __init__(self, args):
        self.args = args
        self.create_pre_process()

    def create_pre_process(self):
        if "east" in args.ngf.lower():
            self.default_shape = [1, 3, 704, 1280]
        elif "sast" in args.ngf.lower():
            self.default_shape = [1, 3, 896, 1536]
        elif "dbnet" in args.ngf.lower():
            self.default_shape = [1, 3, 736, 1280]
        else:
            self.default_shape = [1, 3, 544, 960]

        self.func_load_input = hxocr.load_input_tvm_dbnet
        self.pre_process_list = [
            {"DecodeImage": {"img_mode": "BGR", "channel_first": False}},
            {"DetLabelEncode": None},
            {"DetResizeForTest": {"image_shape": self.default_shape[-2:]}},
            {
                "NormalizeImage": {
                    "std": [0.229, 0.224, 0.225],
                    "mean": [0.485, 0.456, 0.406],
                    "scale": "1./255.",
                    "order": "hwc",
                }
            },
            {"ToCHWImage": None},
            {"KeepKeys": {"keep_keys": ["image", "shape", "polys", "ignore_tags"]}},
        ]

    def inference_aigpu(self, batch_inputs, ngf_file):
        # only support 1 batch
        hx_infer_input_path = "./hx_infer_inputs"
        if os.path.exists(hx_infer_input_path):
            shutil.rmtree(hx_infer_input_path, ignore_errors=True)
        os.mkdir(hx_infer_input_path)

        hx_infer_output_path = "./hx_infer_output"
        if os.path.isdir(hx_infer_output_path):
            shutil.rmtree(hx_infer_output_path, ignore_errors=True)
        os.mkdir(hx_infer_output_path)

        batch_size = batch_inputs.shape[0]
        hx_input_list = []
        for i in range(batch_size):
            hx_input_name = "input" + str(i) + ".bin"
            t = hx_infer_input_path + "/" + hx_input_name
            batch_inputs[i, :, :, :].tofile(t)
            hx_input_list.append(t)
        inputs_str = ":".join(hx_input_list)

        cmd = [
            "hx-infexec",
            "infer",
            "--ngf",
            ngf_file,
            "--batch-size",
            str(batch_size),
            "--batch-count",
            str(1),
            "--inputs",
            inputs_str,
            "--output-dir",
            hx_infer_output_path,
        ]
        sp.run(cmd)

        for i in range(batch_size):
            output_files = [os.path.join(hx_infer_output_path, "job_" + str(i), m) for m in os.listdir(os.path.join(hx_infer_output_path, "job_" + str(i)))]
            out_lists = [np.fromfile(i, dtype=np.float32) for i in output_files]
            out_dicts = {i.shape[0]: i for i in out_lists}
            sorted_out = sorted(out_dicts.items())
            if "east" in ngf_file:
                if len(sorted_out) != 2:
                    raise Error("numer output file error")
                ngf_out1 = sorted_out[1][1].reshape(1, 8, self.default_shape[2] // 4, self.default_shape[3] // 4)
                ngf_out2 = sorted_out[0][1].reshape(1, 1, self.default_shape[2] // 4, self.default_shape[3] // 4)
                ngf_out = [ngf_out1, ngf_out2]
            elif "sast" in ngf_file:
                if len(sorted_out) != 4:
                    raise Error("numer output file error")
                ngf_out1 = sorted_out[2][1].reshape(1, 4, self.default_shape[2] // 4, self.default_shape[3] // 4)
                ngf_out2 = sorted_out[0][1].reshape(1, 1, self.default_shape[2] // 4, self.default_shape[3] // 4)
                ngf_out3 = sorted_out[1][1].reshape(1, 2, self.default_shape[2] // 4, self.default_shape[3] // 4)
                ngf_out4 = sorted_out[-1][1].reshape(1, 8, self.default_shape[2] // 4, self.default_shape[3] // 4)
                ngf_out = [ngf_out1, ngf_out2, ngf_out3, ngf_out4]
            else:
                if len(sorted_out) != 1:
                    raise Error("numer output file error")
                ngf_out = np.fromfile(
                    output_files[0], dtype=np.float32
                )
                ngf_out = ngf_out.reshape(1, 1, self.default_shape[2], self.default_shape[3])

        shutil.rmtree(hx_infer_input_path, ignore_errors=True)
        shutil.rmtree(hx_infer_output_path, ignore_errors=True)
        return ngf_out

    def create_post_process(self, raw_shape):
        if "east" in args.ngf.lower():
            func_post_process = hxocr.east_det_post_process
        elif "sast" in args.ngf.lower():
            func_post_process = hxocr.sast_det_post_process
        else:
            func_post_process = hxocr.dbnet_post_process
        args_post_process = (self.default_shape, raw_shape, True)
        return func_post_process, args_post_process

    def get_raw_shape(self, path):
        h, w, c = cv2.imread(path).shape
        return [1, c, h, w]


def parse():
    desc = "HX arg parser"
    parser = argparse.ArgumentParser(
        description=desc, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--num_val",
        "-nv",
        help="if valid, how many input imgs, default all of dataset",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--ngf",
        "-ngf",
        help="converted engine file",
        type=str,
        default="./en_mv3_dbnet_det_lat.ngf",
    )
    parser.add_argument(
        "--data_path",
        "-dpath",
        help="dataset root path",
        type=str,
        default="./ICDAR2015/",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse()
    ocr_det_eval = DetMetric()
    model_eval = OCRDetectionValid(args)

    dataset_gt_path = os.path.join(args.data_path, "Challenge4_Test_Task1_GT")
    dataset_img_path = os.path.join(args.data_path, "ch4_test_images")
    icdar_labels = ICDAR_utils(dataset_gt_path)
    _nv = args.num_val or hxutils.count_image_in_dir(dataset_img_path)

    for img_name in tqdm(sorted(os.listdir(dataset_img_path))[:_nv]):
        img_path = os.path.join(dataset_img_path, img_name)
        data = {"img_path": img_path, "label": json.dumps(icdar_labels[img_name])}
        data_processes = hxutils.deal_with_variable_args(
            model_eval.func_load_input, data, model_eval.pre_process_list
        )
        data_processes_numpy = [
            np.expand_dims(i.numpy(), axis=0)
            if isinstance(i, paddle.Tensor)
            else np.expand_dims(i, axis=0)
            for i in data_processes
        ]

        output = model_eval.inference_aigpu(data_processes_numpy[0], args.ngf)
        raw_shape = model_eval.get_raw_shape(data["img_path"])

        func_post_process, args_post_process = model_eval.create_post_process(raw_shape)
        ocr_detection = hxutils.deal_with_variable_args(
            func_post_process,
            output,
            args_post_process,
        )
        ocr_det_eval(ocr_detection, data_processes_numpy)
    
    metric = ocr_det_eval.get_metric()
    # metric['fps'] = total_frame / total_time
    print("accuracy:")
    print("==" * 50)
    for k, v in metric.items():
        print("{}:{}".format(k, v))
