# Convert A en_v3_det Model in Paddle fmk Samples
- Function：graph compilation, graph optimization and format conversion for en_v3_det model in Paddle fmk
- Input：en_v3_det model in  Paddle fmk
- Output：en_v3_det model in ngf format

## Prepare
| environment | requirement | Notes |
|---|---|---|
|Hardware|Servers with Hexaflake C10 PCIE board|Servers please refer to [servers support list]|
|Software|TOCA docker released by Hexaflake|Please refer to install document|

## Samples

1. Get pub-models

    ```
    Please contact Hexaflake technical support to get pub-models
    ```
2. Model Convert

    | **Model Name** | **Category** | **Framework** | **Download** |
    |---|---|---|---|
    | en_v3_det | detection | Paddle | https://modelzoo.hexaflake.com/ocr_detection/paddle/en_v3_det/en_v3_det.pdiparams  https://modelzoo.hexaflake.com/ocr_detection/paddle/en_v3_det/en_v3_det.pdmodel|

    **Note**
    - Original Model: https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar

    ```
    ./model_convert.sh

    ```
3. Performance Test

    ```
    ./run_perf.sh --batch_size N # N = 1, 4, 8, 16, 32, 64, 128

    ```