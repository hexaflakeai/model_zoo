#!/bin/bash

dataset_key="ILSVRC2012_img_val"
origin_dataset="ILSVRC2012_img_val"
dataset_name="ILSVRC2012_img_val"

function download_dataset() {
    echo "Dataset ${dataset_name} does not exist. Need to preprocess dataset ..."
    if [[ -v local_dataset && -d ${local_dataset} ]]; then
        origin_dataset=${local_dataset}
        if [ ! -d "${origin_dataset}" ]; then
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

    if [ ! -d ${origin_dataset} ]; then
        mkdir ${origin_dataset}
    fi
    tar_name="${origin_dataset}.tar"
    tar -xvf ${tar_name} -C ${origin_dataset}
    if [ $? -ne 0 ]; then
        echo "ERROR: Unable to extract dataset from ${tar_name}."
        return 6
    fi
    return 0
}

if [ ! -d "${dataset_name}" ]; then
    download_dataset
    if [ $? -ne 0 ]; then
        exit $?
    fi
else
   echo "INFO: Found existing dataset ${dataset_name}."
fi

if [ -v PUB_MODELS_HOME ]; then
    preprocess_path="${PUB_MODELS_HOME}/models/validation/python"
else
    preprocess_path="../../../validation/python"
fi

if [ ! -d ${preprocess_path} ]; then
    echo "ERROR: Cannot find preprocess script at ${preprocess_path}."
    exit 7
fi

# set paddle env
export LD_LIBRARY_PATH=/lib64/stdc:$LD_LIBRARY_PATH

# data preprocess
python3 paddle_data_preprocess.py -img ./${dataset_name} -label ${preprocess_path}/hx-inet50k-label.txt