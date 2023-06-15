# Convert A Vgg19 Model in Paddle Format Samples
- Function：graph compilation, graph optimization and format conversion for vgg19 model in Paddle format
- Input： vgg19 model in Paddle format
- Output： vgg19 model in ngf format

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
    | vgg19 | classification | Paddle | https://modelzoo.hexaflake.com/classification/paddle/vgg19/vgg19.pdiparams  https://modelzoo.hexaflake.com/classification/paddle/vgg19/vgg19.pdmodel |

    **Note**
    - Original Model: https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/VGG19_infer.tar
    ```
    ./model_convert.sh

    ```
3. Performance Test

    ```
    ./run_perf.sh --batch_size N  # N = 1, 2, 4, 8, 16, 32, 64, 128
    ```

4. DataSet Validation

   ```
   ./model_valid.sh
   ``` 
