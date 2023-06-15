import numpy as np
import os, re
from tvm.contrib.hexaflake.tools.logger import hxlog
from tvm.contrib.hexaflake.tools import utils as hxutils
import subprocess
import argparse
import logging


class HXInferExecHelper:
    def __init__(self, args):
        # copy all members of args (tools/argparser) to myself
        for _arg in args._get_kwargs():
            value = f'"{_arg[1]}"' if isinstance(_arg[1], str) else _arg[1]
            exec(f"self.{_arg[0]} = {value}")
        if self.info_dir is None:
            self.info_dir = self.input_dir
        self.input_paths = []

    @staticmethod
    def lower_to_pow2(n):
        res = 0
        for i in range(n, 0, -1):
            if (i & (i - 1)) == 0:
                res = i
                break
        return res

    @staticmethod
    def guess_output_node(path):
        files = os.listdir(path)
        if 0 < len(files) <= 2:
            return files[0]
        else:
            hxlog.error("can not guess which node is the desired output")

    @staticmethod
    def get_max_arg_available(func_path, preserve: int = 100):
        # TODO: not working
        try:
            arg_max = int(os.popen("getconf _POSIX_ARG_MAX").read().strip())
        except:
            arg_max = 8192
        finally:
            arg_max = min(arg_max, 32767)
        try:
            used = int(os.popen("env | wc -c").read().strip())
        except:
            used = 1000
        return arg_max - used - len(func_path) - 4096

    def prepare_batched_input(self):
        self.ngf_batch_num = -(-self.num_val // self.ngf_batch_size)  # up rounding
        dir_tmp_input = os.path.join(
            self.temp_dir, f".hexaflake/batched_input/{self.input_dir}"
        )
        hxutils.empty_dir(dir_tmp_input)
        for n in range(self.ngf_batch_num):
            fname_batched = f"{self.input_prefix}{n}{self.input_suffix}"
            fpath_batched = os.path.join(dir_tmp_input, fname_batched)
            with open(fpath_batched, "wb") as dst:
                for b in range(self.ngf_batch_size):
                    fnum = n * self.ngf_batch_size + b
                    fnum = fnum if fnum <= self.num_val else fnum - 1
                    fname = f"{self.input_prefix}{fnum}{self.input_suffix}"
                    fpath = os.path.join(self.input_dir, fname)
                    with open(fpath, "rb") as src:
                        dst.write(src.read())
        self.input_dir_batched = dir_tmp_input
        self.num_val_batched = self.ngf_batch_num

    def prepare_input_paths(self, prefix: str = "input-", suffix: str = ".bin"):

        self.input_prefix = prefix
        self.input_suffix = suffix
        num_input_aval = len(os.listdir(self.input_dir))
        assert (
            self.num_val <= num_input_aval
        ), f"number of validation must less or equal than the existed {num_input_aval}"
        # set zero to use all
        self.num_val = self.num_val or num_input_aval

        # setup batched input dir and paths
        if self.ngf_batch_size != 1:
            self.prepare_batched_input()
            self.inferexec_num_val = self.num_val_batched
            self.inferexec_input_dir = self.input_dir_batched
        else:
            self.inferexec_num_val = self.num_val
            self.inferexec_input_dir = self.input_dir

        for i in range(self.inferexec_num_val):
            self.input_paths.append(
                os.path.join(self.inferexec_input_dir, f"{prefix}{i}{suffix}")
            )

    def run_validation(self):
        """
        what I have:
            self.input_paths = paths of bin file _0.bin _1.bin ...
            self.inferexec = path to infexec
            self.ngf_path = ngf_path file
        what I need to generate:
            predictions to self.output_dir, name as pred_0.bin, _1.bin ...
        more info:
            1. calc the path length to take input as much as I can
            2. calc the file size for maximum DDR utilization
            3. maximun length of linux cmd: 8191 chars
        infexec example:
            hx-infexec infer --ngf_path ./bert-base.ngf_path --batch-size N --batch-count 1000 \
                                --inputs ./input_data/0_input_ids-0-1_128_HW32w,\
                                ./input_data/1_input_type_ids-0-1_128_HW32w,\
                                ./input_data/2_input_mask-0-1_128_HW32w
        """
        # calculate how many inputs can be bonded
        CLI_MAX_LENGTH = self.get_max_arg_available(self.inferexec)
        INFERSER_MAX_BATCH = os.getenv("INFERSER_MAX_BATCH", 128)
        self.inferexec_batch_count = 1
        self.inferexec_batch_size = min(
            INFERSER_MAX_BATCH,
            len(self.input_paths),
            self.lower_to_pow2(CLI_MAX_LENGTH // len(self.input_paths[0])),
        )
        self.inferexec_batch_number = -(
            -self.inferexec_num_val // self.inferexec_batch_size
        )
        # do infer for batchs
        for b in range(self.inferexec_batch_number):
            self.inplist = ""
            for i in range(self.inferexec_batch_size):
                idx = b * self.inferexec_batch_size + i
                if idx >= min(self.inferexec_num_val, len(self.input_paths)):
                    break
                self.inplist += f":{self.input_paths[idx]}"
            hxlog.progress(
                f"doing inference by hx-inferexec, batch size: {self.inferexec_batch_size}",
                b,
                self.inferexec_batch_number,
            )
            self.drive_inferexec(b)

    def drive_inferexec(self, b):
        inplist = self.inplist
        cmd = [
            self.inferexec,
            "infer",
            "--ngf",
            self.ngf_path,
            "--batch-size",
            str(self.inferexec_batch_size),
            "--batch-count",
            str(self.inferexec_batch_count),
            "--inputs",
            inplist[1:] if inplist.startswith(":") else inplist,
            "--output-dir",
            hxutils.empty_dir(os.path.join(self.output_dir, str(b))),
            "--device",
            str(self.device),
        ]

        str_cmd = " ".join(cmd)
        hxlog.debug(str_cmd)
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            stdout, stderr = proc.communicate()
            stdout = stdout.decode().replace("\\n", "\n")
            stderr = stderr.decode().replace("\\n", "\n")
        except Exception as e:
            proc = lambda: None
            proc.returncode = -1
            stdout = None
            stderr = e
        finally:
            if proc.returncode != 0:
                hxlog.error(
                    f"error occurred when doing inference with hx-infexec CLI.\n"
                    + f"return message:\n  stdout: {stdout}\n  stderr: {stderr}\n"
                    + f"use the CLI command below for debugging:\n{str_cmd}"
                )

    def get_label_data(self, i, prefix: str = "info-", suffix: str = ".npy"):
        data = np.load(
            os.path.join(self.info_dir, f"{prefix}{i}{suffix}"), allow_pickle=True
        ).item()
        return data["label"]

    def get_detection_info_dict(self, i, prefix: str = "info-", suffix: str = ".npy"):
        data = np.load(
            os.path.join(self.info_dir, f"{prefix}{i}{suffix}"), allow_pickle=True
        ).item()
        # TODO: multi-static-batch validation
        self.input_infos = {"batched_info": [hxutils.dict_load_and_pop(data, "infos")]}
        self.ground_truth = data

    def get_facenet_meta(self, dir, file="facenet_eval_info.npy"):
        path = os.path.join(dir, file)
        data = np.load(path, allow_pickle=True).item()
        self.facenet_meta = lambda: None
        for k in data:
            self.facenet_meta.__setattr__(f"{k}", data[k])

    def split_aigpu_output(self):
        splited = []
        for i, _output in enumerate(self.outputs):
            _output = _output[: np.prod(self.output_shapes[i] + [self.ngf_batch_size])]
            _output = _output.reshape(
                [self.ngf_batch_size, np.prod(self.output_shapes[i])]
            )
            _output = [
                d.reshape(self.output_shapes[i])
                for d in np.split(_output, self.ngf_batch_size)
            ]
            splited.append(_output)
        self.outputs = splited

    def load_single_output(self, path):
        with open(path, "r") as f:
            output = np.fromfile(f, self.np_dtype)
        if self.data_type == "bfloat16":
            output = (output.astype(np.int32) << 16).view(np.float32)
        if self.output_offset >= 0:
            return output[self.output_offset :]
        else:
            return output[: self.output_offset]

    def load_aigpu_output_single(self, b, n, prefix: str = "job_"):
        fdir = os.path.join(self.output_dir, str(b), f"{prefix}{n}")

        if self.output_nodes is None:
            self.output_nodes = self.guess_output_node(fdir)

        self.np_dtype = np.uint16 if self.data_type == "bfloat16" else np.float32

        if not isinstance(self.output_nodes, list):
            self.output_nodes = [self.output_nodes]
        self.outputs = []
        for node in self.output_nodes:
            fpath = os.path.join(fdir, node)
            self.outputs.append(self.load_single_output(fpath))

    def load_aigpu_output(self, b, n, prefix: str = "job_"):
        self.load_aigpu_output_single(b, n, prefix)
        self.split_aigpu_output()
        return self.outputs

    def clean_output(self):
        hxutils.empty_dir(self.output_dir)


def argparser():
    def _parse_nodes(v):
        delimiter = [" ", ","]
        if any([_d in v for _d in delimiter]):
            l = re.split("|".join(delimiter), v)
            while "" in l:
                l.remove("")
            return l
        else:
            return v

    # fmt: off
    desc = "HX inferexec helper parser"
    parser = argparse.ArgumentParser(description=desc, formatter_class=argparse.RawTextHelpFormatter)
    # paths
    parser.add_argument("--inferexec", "-exec", help="path to hexaflake inferexecutor", type=str, default="hx-infexec")
    parser.add_argument("--ngf-path", "-ngf", help="path to aigpu engine file, use the hx-convert generated *.ngf", type=str, required=True, default=None)
    parser.add_argument("--input-dir", "-input", help="directory to find the input data", type=str, required=True, default="./input")
    parser.add_argument("--info-dir", "-info", help="directory to find the info data that contains ground truth for validation", type=str, default=None)
    parser.add_argument("--output-dir", "-output", help="directory to put inference output", type=str, default="./output")
    parser.add_argument("--temp-dir", "-temp", help="path to store temp files", type=str, default="./temp")
    # conf
    parser.add_argument("--ngf-batch-size", "-nb", help="batch size inside ngf", type=int, default=1)
    parser.add_argument("--num-val", "-nv", help="if valid, how many inputs for validation, default: 10, set 0 to valid on full dataset", type=int, default=10)
    parser.add_argument("--data-type", "-dtype", help="inference datatype, it may help with the output decoding, default: float32", type=str, default="float32")
    parser.add_argument("--output-nodes", "-nodes", help="output node for post processing, will guess as default, input example: \"node_name0.dat, node_name_1.dat\"", type=_parse_nodes, default=None)
    # for classification
    parser.add_argument("--output-offset", "-offset", help="offset to find the 1000 output of classification model, genereally 0, if the raw model output 1001 outputs, try one", type=int, default=0)
    parser.add_argument("--label-order", "-order", help="label order for validation: [WordNet/ImageNet], typically tensorflow/pytorch adopts WordNet order, default: WordNet", type=str, default="WordNet")
    # for detection
    parser.add_argument("--input-size", "-size", help="model input size for detection postprocess, default: None, will use default settings in the valid script", type=int, default=None)
    parser.add_argument("--model", "-model", help="model name for detection postprocess, default: None, will guess base on the ngf file name", type=str, default=None)
    parser.add_argument("--annotation-file", "-ann", help="annotation file for COCO dataset, default: ./instances_val2017.json", type=str, default="./instances_val2017.json")
    parser.add_argument("--device", "-dev", help="select a device to execute the inference on multi-card host, default: 0", type=int, default=0)
    # for debugging
    parser.add_argument("-v", "--verbose", action="count", default=0, help="increase verbosity.")
    args = parser.parse_args()
    hxlog.set_level(hxlog.INFO - args.verbose * 10) # TODO: set the logging level with logging module
    return args
    # fmt: on
