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
def main_func(name, path, data_folder):
    input_name = "inputs"
    default_shape = [1, 3, 704, 1280]
    model = HXModelPaddle(hxargs, name, path, "x", default_shape)
    model.func_load_input = hxocr.load_input_tvm_dbnet
    model.args_load_input = model.pre_process_list

    model.func_post_process = hxocr.east_det_post_process
    model.func_get_output = hxcomm.get_output_general
    model.args_get_output = (model.target, model.tvm_output_dtype, model.dev_id)
    model.mod_compile = model.mod
    model.compile_model(None)

    os.makedirs("./out", exist_ok=True)
    imgs = os.listdir(data_folder)
    imgs = imgs if hxargs.num_val > len(imgs) else imgs[: hxargs.num_val]
    for i in tqdm(imgs):
        img_path = os.path.join(data_folder, i)
        raw_h, raw_w, raw_c = cv2.imread(img_path).shape
        raw_shape = [1, raw_c, raw_h, raw_w]
        model.args_post_process = (model.input_shape, raw_shape)

        tvm_predictions = model.predict(img_path)
        src_im = hxocr.draw_text_det_dbnet(tvm_predictions, img_path)
        img_path = os.path.join("./out", "det_res_{}".format(i))
        cv2.imwrite(img_path, src_im)


def main():
    hxargs.framework = hxargs.framework or "paddle"
    model = hxargs.model or "./en_mv3_east_det/"
    hxargs.category = "ocr_detection"
    data_folder = hxargs.path_to_inet_root
    model_name = model.split("/")[-2] if model.endswith("/") else model.split("/")[-1]
    return main_func(model_name, os.path.join(model, model_name), data_folder)


if __name__ == "__main__":
    hxargs = HXargparser().args
    hxlog.set_level("error")
    exit(main())
