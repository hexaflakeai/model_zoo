#!/bin/bash
model_name="mobilenetv1"
base_batch_count=1000
batch=8
cur_path=$(cd $(dirname $0);pwd)
model_home=${cur_path}

declare -i runPerfError=1

function infer() {
    # infer latency model
    if [[ ${batch} -eq 1 ]]; then
        echo "INFO: Using minimum latency engine file for single batch size inference."
        hx-infexec infer --ngf ${model_name}_lat.ngf --batch-count 1000 --batch-size ${batch} ${infer_duration} ${infer_device} ${output_buffer}
    else
        echo "INFO: Inference is using maximum throughput engine file for batch size ${batch}"
        hx-infexec infer --ngf ${model_name}.ngf --batch-count 1000 --batch-size ${batch} ${infer_duration} ${infer_device} ${output_buffer}
    fi
    if [[ $? -ne 0 ]]; then
        return 1
    fi
}

while true; do
  case "$1" in
    --batch_size )
      batch="$2"; shift 2 ;;
    --duration)
      infer_duration="--duration=$2"; shift 2 ;;
    --device)
      infer_device="--device=$2"; shift 2 ;;
    --output_buffer)
      output_buffer="--output-buffer $2"; shift 2 ;;
    * ) break ;;
  esac
done

if [[ ${batch} -le 0 ]]; then
    echo "ERROR: Illegal batch size ${batch}."
    exit 2
fi

if [[ $batch == 1 ]]; then
    echo "batch size 1"
    if [ ! -f ${model_name}_lat.ngf ]; then
        echo "ERROR: Cannot find ${model_name}_lat.ngf. Please run model_convert.sh firstly."
        exit 3
    fi
    infer ${model_name}_lat.ngf  1
else
    if [ ! -f ${model_name}.ngf ]; then
        echo "ERROR: Cannot find ${model_name}.ngf. Please run model_convert.sh firstly."
        exit 4
    fi

    infer ${model_name}.ngf  4
    if [ $? -ne 0 ]; then
        echo "ERROR: Inference failed."
        exit 5
    fi
    echo "INFO: Inference finished."
fi