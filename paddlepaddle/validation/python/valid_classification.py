import os
import inferexec as inferexec
from tvm.contrib.hexaflake.tools.logger import hxlog
from tvm.contrib.hexaflake.vision import classification as hxclass

PWD = os.path.dirname(__file__)


def main():
    hxargs = inferexec.argparser()

    # prepare model, run validation
    model = inferexec.HXInferExecHelper(hxargs)
    model.prepare_input_paths(prefix="input-", suffix=".bin")
    model.run_validation()
    # prepare ground truth, check result
    # TODO: for pub-model publishing, replace the following path, and remove this comment
    label_path = os.path.join(PWD, "hx-class-id-label-names.txt")
    gt_path = os.path.join(PWD, "hx-inet50k-label.txt")
    hxlabel = hxclass.HXClassLabel(model.label_order, label_path, gt_path)
    accuracy = hxclass.HXClassAccuracy()
    model.output_shapes = [[1000]]
    for b in range(model.inferexec_batch_number):
        for n in range(model.inferexec_batch_size):
            for nb in range(model.ngf_batch_size):
                i = (
                    b * model.inferexec_batch_size * model.ngf_batch_size
                    + n * model.ngf_batch_size
                    + nb
                )
                if i >= model.num_val:
                    break
                if nb == 0:
                    outputs = model.load_aigpu_output(b, n, prefix="job_")
                output = outputs[0][nb]  # class model have only one output
                _, top5idx = hxclass.topk(output, 5)
                pred_lab = hxlabel.get_pred_label(top5idx)
                gtlabel = model.get_label_data(i)
                accuracy.update(gtlabel, pred_lab)
                hxlog.progress(
                    f"num: {accuracy.neval}, top1: {accuracy.top1:.4%}, top5: {accuracy.top5:.4%}",
                    i,
                    model.num_val,
                )
    model.clean_output()


if __name__ == "__main__":
    main()

# example:
# python tests/hexaflake/pub-model/valid_classification.py -ngf "./torch_vgg16_float32.ngf" -input "./input" -info "./info" -out "./output" -nv 50000 -offset 0
