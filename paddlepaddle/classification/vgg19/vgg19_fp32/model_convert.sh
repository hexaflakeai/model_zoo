#!/bin/bash
model_class="classification"
model_name="vgg19"
dtype="fp32"
infer_dtype="float32"
model_ver=""
frame_work="paddle"
format=(".pdiparams" ".pdmodel")
model_key="vgg19_paddle"

model_file_name=("${model_name}${model_ver}${format[0]}" "${model_name}${model_ver}${format[1]}")
thr_engine_file="${model_name}${model_ver}.ngf"
lat_engine_file="${model_name}${model_ver}_lat.ngf"

script_path="$( cd "$(dirname $BASH_SOURCE)" ; pwd -P)"
project_path=${script_path}

function download_model() {
    echo "INFO: Model file ${model_file_name[@]} does not exist. Downloading ..."
    if [ -v model_url_json ]; then
        echo "INFO: Using links in ${model_url_json}"
    elif [ -v PUB_MODELS_HOME ]; then
        model_url_json=${PUB_MODELS_HOME}/model_url.json
        echo "INFO: Using links in ${PUB_MODELS_HOME}/model_url.json"
    else
        model_url_json=../../../../model_url.json
    fi

    urls=()
    find_key=$(cat ${model_url_json} | jq 'has ("'${model_key}'")')
    if [ -f ${model_url_json} ] && [ "${find_key}" == "true" ]; then
        index=0
        while true
        do
            url=$(jq -r ".${model_key}[$index]" $model_url_json)
            if [[ $url == null ]]; then
                break;
            else
                urls+=${url}" "
            fi
            let index=$index+1
        done
    else
        echo "WARNING: Cannot find model_url.json at ${model_url_json} or cannot find the model key ${model_key} in it. Try default link ..."
        for f in ${model_file_name}; do
            rel_path="${model_class}/${model_name}/${frame_work}/${model_name}_${dtype}/${f}"
            if [ -v external_zoo_site ]; then
                echo "INFO: Using defined modelzoo at ${external_zoo_site}"
                urls+="${external_zoo_site}/${rel_path}" ""
            else
                urls+="https://modelzoo.hexaflake.com/${rel_path}" ""
            fi
        done
    fi
    for url in ${urls}; do
        curl --connect-timeout 3 -IL -f ${url}
        if [[ $? -ne 0 ]]; then
            echo "WARNING: Failed to download model from ${url}. Try backup link ..."
            if [[ ${url} == "https://zoo/"* ]]; then
                url=${url/"https://zoo/"/"https://modelzoo.hexaflake.com/"}
            else
                url=${url/"https://modelzoo.hexaflake.com/"/"https://zoo/"}
            fi
            curl -# ${url} -o "${model_name}/${url##*/}" --create-dir
            if [[ $? -ne 0 ]]; then
                echo "ERROR: Failed to download model, please check if the link is accessible."
                return 1
            else
                curl -# ${url} -o "${model_name}/${url##*/}" --create-dir
                if [[ $? -ne 0 ]]; then
                    echo "ERROR: Failed to download model, please check if the link is accessible."
                    return 1
                fi
            fi
        else
            curl -# ${url} -o "${model_name}/${url##*/}" --create-dir
            if [[ $? -ne 0 ]]; then
                echo "ERROR: Failed to download model, please check if the link is accessible."
                return 1
            fi
        fi
    done

    return 0
}

function preprocess() {
    model_file_name=${model_file_name[0]}
}

function main() {
    need_download=0
    for f in ${model_file_name}; do
        if [ ! -f "resnet50_paddle/${f}" ] ; then
            need_download=1
            break
        fi
    done
    if [ ${need_download} -eq 1 ]; then
        # remove incomplete model files
        for f in ${model_file_name}; do
            if [ ! -f ${f} ]; then
                rm -f ${f}
            fi
        done
        # get original model file
        download_model
        if [ $? -ne 0 ];then
            return 1
        fi
    else
        echo "INFO: Use existing ${model_file_name[@]} ..."
    fi

    preprocess
    if [ $? -ne 0 ];then
        return 2
    fi

    # convert paddle model to engine file
    export LD_LIBRARY_PATH=/lib64/stdc:${LD_LIBRARY_PATH}
    hx-convert compile --target "aigpu" \
    --model-format ${frame_work} \
    --output ${lat_engine_file} \
    "${model_name}/${model_name}.pdmodel" \
    --input-shapes "inputs:[1,3,224,224]" \
    --latency
    if [ $? -ne 0 ];then
        echo "ERROR: Failed to optimize the model for the minimum latency."
        return 3
    fi

    hx-convert compile --target "aigpu" \
    --model-format ${frame_work} \
    --output ${thr_engine_file} \
    "${model_name}/${model_name}.pdmodel" \
    --input-shapes "inputs:[1,3,224,224]"
    if [ $? -ne 0 ];then
        echo "ERROR: Failed to optimize the model for the maximum throughput."
        return 3
    fi
}

main
