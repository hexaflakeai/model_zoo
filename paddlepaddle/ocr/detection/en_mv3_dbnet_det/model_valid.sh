#!/bin/bash
model_class="paddle"
model_name="en_mv3_dbnet_det"
dtype="fp32"
model_ver=""
dataset_key="ICDAR2015"
engine_file="${model_name}${model_ver}_lat.ngf"
dataset_name="ICDAR2015"
model_file_name=("${model_name}${model_ver}${format}")
num_of_inputs=0

while true; do
  case "$1" in
    --num_of_inputs)
      num_of_inputs="$2"; shift 2 ;;
    --device)
      infer_device="--device=$2"; shift 2 ;;
    * ) break ;;
  esac
done

if [ ! -f ${engine_file} ]; then
    echo "ERROR: Cannot find ${engine_file}. Please run model_convert.sh firstly."
    exit 2
fi

function download_dataset() {
    echo "Dataset ${dataset_name} does not exist. Downloading ..."
    if [[ -v local_dataset && -d ${local_dataset} ]]; then
        dataset_name=${local_dataset}
        if [ ! -d "${dataset_name}" ]; then
            echo "ERROR: Dataset given by '${local_dataset}='${local_dataset} does not exist."
            return 3
        fi
        return 0
    fi
    if [ -v dataset_url_json ]; then
        echo "INFO: Using links in ${dataset_url_json}"
    elif [ -v PUB_MODELS_HOME ]; then
        dataset_url_json=${PUB_MODELS_HOME}/model_url.json
        echo "INFO: Using links in ${PUB_MODELS_HOME}/model_url.json"
    else
        dataset_url_json=../../../../model_url.json
    fi

    find_key=$(cat ${dataset_url_json} | jq 'has ("'${dataset_key}'")')
    if [ -f ${dataset_url_json} ] && [ "${find_key}" == "true" ]; then
        url=$(jq -r ".${dataset_key}[0]" $dataset_url_json)
    else
        echo "ERROR: Cannot find the URL for dataset ${dataset_key} or cannot find the dataset key ${dataset_key} in it."
        return 4
    fi
    curl --connect-timeout 3 -IL -f ${url}
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to download dataset, please check if the link is accessible."
        return 5
    else
        curl -# -O ${url}
        if [ $? -ne 0 ]; then
            echo "ERROR: Failed to download dataset, please check if the link is accessible."
            return 5
        fi
    fi

    tar_name="${dataset_name}.tar.gz"
    tar -xvf ${tar_name}
    if [ $? -ne 0 ]; then
        echo "ERROR: Unable to extract dataset from ${tar_name}."
        return 6
    fi
    return 0
}

if [ ! -d "${dataset_name}" ]; then
    download_dataset
    if [ $? -ne 0 ]; then
        exit 1
    fi
else
   echo "INFO: Found existing dataset."
fi

export LD_LIBRARY_PATH=/lib64/stdc:${LD_LIBRARY_PATH}
python3 ../../../validation/python/valid_ocr_detection.py -ngf ${engine_file} -dpath ./ICDAR2015 -nv ${num_of_inputs}
