from tqdm import tqdm
import time, re, os, cv2
import tvm.contrib.hexaflake.tools.common as hxcomm
import tvm.contrib.hexaflake.vision.ocr as hxocr
import tvm.contrib.hexaflake.vision.detection as hxdetect
import tvm.contrib.hexaflake.vision.classification as hxclass
from tvm.contrib.hexaflake.tools.logger import hxlog
from tvm.contrib.hexaflake.tools.argparser import HXargparser
from tvm.contrib.hexaflake.tools.quantization import QCONF_DEFAULT
from tvm.contrib.hexaflake.driver.paddle import HXModelPaddle


@hxlog.logit
def get_labels():
    out_labels = {}
    with open(hxargs.path_to_inet_label) as f:
        labels = f.readlines()

    for line in labels:
        cls_id = line.strip().split()[0]
        cls_name = " ".join(line.strip().split()[1: ])
        out_labels[int(cls_id)] = cls_name
    return out_labels


@hxlog.logit
def main_func(name, path, data_folder, hxargs):
    input_name = "inputs"
    default_shape = [1, 3, 224, 224]
    model = HXModelPaddle(hxargs, name, path, input_name, default_shape)
    model.func_load_input = hxcomm.load_input_tvm_paddle_classification

    model.func_post_process = hxclass.paddle_classification_post_process
    model.args_post_process = (model.topk, model.class_id_map_file)

    model.args_load_input = model.pre_process_list
    model.func_get_output = hxcomm.get_output_general
    model.args_get_output = (model.target, model.tvm_output_dtype, model.dev_id)

    model.mod_compile = model.mod
    model.compile_model(None)

    imgs = [os.path.join(data_folder, i) for i in os.listdir(data_folder)]
    imgs = imgs if hxargs.num_val > len(imgs) else imgs[: hxargs.num_val]
    
    labels = get_labels()

    with open("predict.txt", "w") as f:
        for i in tqdm(imgs):
            tvm_predictions = model.predict(i)
            if hxargs.category == "classification":
                f.write(f"img_path: {i} class_name: {labels.get(tvm_predictions[1][0])} conf: {tvm_predictions[0][0]}\n")
                print(f"img_path: {i} class_name: {labels.get(tvm_predictions[1][0])} conf: {tvm_predictions[0][0]}\n")
            

def main():
    hxargs.framework = hxargs.framework or "paddle"
    print("*"*100)
    model = hxargs.model or "./mobilenetv2/"
    hxargs.category = "classification"
    data_folder = hxargs.path_to_inet_root
    model_name = model.split("/")[-2] if model.endswith("/") else model.split("/")[-1]
    return main_func(model_name, os.path.join(model, model_name), data_folder, hxargs)


if __name__ == "__main__":
    hxargs = HXargparser().args
    hxlog.set_level("error")
    exit(main())
