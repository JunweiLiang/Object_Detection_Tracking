# Experiments on Inferencing Speed



## Experiment 1: All runs get almost exactly the same outputs

Machine 1: 2 GTX 1080 TI, i7, nvme

Machine 2: 3 GTX 1080 TI + 1 TITAN X, E5, nvme

Machine 3: 4 RTX 2080 TI , i9-9900X, SSD

Each run is conducted without other programs running except \*.

Machine 1

| RunType | # Image | Image Size | # GPU Used | runtime (s) | GPU Average Utilization | Per GPU FPS |
|---------|---------|------------|------------|-------------|-------------------------|-------------|
| 1       | 206268  | 1920x1080  | 2          | 41190.9     | 65.3%                   | 2.50        |
| 2       | 206268  | 1920x1080  | 2          | 37920.9     | 62.0%                   | 2.72        |
| 3       | 206268  | 1920x1080  | 2          | 31494.2     | 54.8%                   | 3.27        |
| 2       | 206268  | 1920x1080  | 1          | 75652.2     | 70.1%                   | 2.73        |
| 3       | 206268  | 1920x1080  | 1          | 57484.9     | 68.8%                   | 3.59        |

Machine 2

| RunType | # Image | Image Size | # GPU Used | runtime (s) | GPU Average Utilization | Per GPU FPS |
|---------|---------|------------|------------|-------------|-------------------------|-------------|
| 2       | 206268  | 1920x1080  | 4          | 29354.4     | 33.8%                   | 1.76        |
| 3       | 206268  | 1920x1080  | 4          | 26454.0     | 23.5%                   | 1.95        |
| 3       | 206268  | 1920x1080  | 2          | 35975.7     | 38.0%                   | 2.87        |
| 2       | 206268  | 1920x1080  | 1          | 74065.0     | 41.4%                   | 2.78        |
| 3       | 206268  | 1920x1080  | 1          | 57880.8     | 54.8%                   | 3.56        |
| 2       | 206268  | 1920x1080  | 4 / 1*     | 17556.3     | 46.2%                   | 2.94        |
| 3       | 206268  | 1920x1080  | 4 / 1*     | 14552.5     | 52.3%                   | 3.54        |
| 5       | 206268  | 1920x1080  | 4 / 1*     | 17027.5     | 53.2%                   | 3.03        |
| 6       | 206268  | 1920x1080  | 4 / 1*     | 13433.3     | 61.7%                   | 3.84        |


Machine 3

| RunType | # Image | Image Size | # GPU Used | runtime (s) | GPU Average Utilization | Per GPU FPS |
|---------|---------|------------|------------|-------------|-------------------------|-------------|
| 2       | 206268  | 1920x1080  | 1          | 57834.6     | 61.2%                   | 3.57        |
| 3       | 206268  | 1920x1080  | 1          | 43510.9     | 61.2%                   | 4.74        |
| 2       | 206268  | 1920x1080  | 4 / 1*     | 14143.3     | 62.6%                   | 3.65        |
| 3       | 206268  | 1920x1080  | 4 / 1*     | 10676.3     | 65.2%                   | 4.83        |

TODO: Add input queue mechanism to improve GPU utilization.

| RunType |                                                                         |
|---------|-------------------------------------------------------------------------|
| 1       | tf 1.10 (CUDA 9, cudnn 7.1), Variable Model                             |
| 2       | tf 1.13 (CUDA 10.0 cudnn 7.4), Variable Model                           |
| 3       | tf 1.13 (CUDA 10.0 cudnn 7.4), Frozen Graph (.pb)                       |
| 4       | tf 1.13 (CUDA 10.0 cudnn 7.4), Frozen Graph (.pb) -> TensorRT Optimized |
| 5       | tf 1.14.0 (CUDA 10.0 cudnn 7.4), Variable Model                         |
| 6       | tf 1.14.0 (CUDA 10.0 cudnn 7.4), Frozen Graph (.pb)                     |


4 / 1 * means that I run 4 single-gpu jobs in parallel, just the same as you would run in the full system.


Conclusions:
- TF v1.10 -> v1.13 (CUDA 9 & cuDNN v7.1 -> CUDA 10 & cuDNN v7.4) ~ +9% faster
- Use frozen graph  ~ +30% faster
- GTX 1080 TI -> RTX 2080 TI ~ +30% faster

Note that I didn't have time to run these experiments repeatly so I expect the numbers to have large variances.

To freeze the model into a .pb file:
```
$ python main.py nothing nothing --mode pack --pack_model_path obj_v3.pb \
--pack_modelconfig_path obj_v3.config.json --load_from obj_v3_model/ --note obj_v3 --num_class 15 \
--diva_class3 --rpn_batch_size 256 --frcnn_batch_size 512 --rpn_test_post_nms_topk 1000 --is_fpn \
--use_dilation --max_size 1920 --short_edge_size 1080
```

Run testing on the v1-val set:
```
$ python main.py nothing v1-validate_frames.lst --mode forward --outbasepath \
obj_v3_val_output --num_class 15 --diva_class3 --max_size 1920 --short_edge_size \
1080 --gpu 2 --im_batch_size 2 --load_from obj_v3.pb  --is_load_from_pb --log
```
Assuming `v1-validate_frames.lst` contains absolute path of all images. This will output one json in COCO detection format for each image in `obj_v3_val_output/`. The `--log` will run `nvidia-smi` every couple second in a separate thread to record the gpu utilizations.

## Experiment 2: TensorRT Optimization (TF v1.14.0)

Use TensorRT to optimize frozen graph (I tried FP32, FP16):
```
$ python tensorrt_optimize.py obj_v5_tfv1.14.0.pb obj_v5_tfv1.14.0.tensorRT.fp32.pb --precision_mode FP32
```

Inferencing (run 4 program in parallel):
```
$ python main.py nothing v1-validate_frames.1.lst --mode forward --outbasepath \
obj_v5_val_output_TRT_FP32 --num_class 15 --diva_class3 --max_size 1920 --short_edge_size \
1080 --gpu 1 --im_batch_size 1 --gpuid_start 0 --load_from obj_v5_tfv1.14.0.tensorRT.fp32.pb  --is_load_from_pb --log
```

I haven't explored `--maximum_cached_engines` and `INT8` mode yet. And ideally these experiments should be repeated a couple of times.

Experiments:

| RunType | Model: obj_v5                                                           |
|---------|-------------------------------------------------------------------------|
| 1       | tf 1.14.0 (CUDA 10.0 cudnn 7.4), Frozen Graph (.pb)                     |
| 2       | tf 1.14.0 (CUDA 10.0 cudnn 7.4), Frozen Graph (.pb) -> TRT FP32         |
| 3       | tf 1.14.0 (CUDA 10.0 cudnn 7.4), Frozen Graph (.pb) -> TRT FP16         |

Machine 2

| RunType | # Image | Image Size | # GPU Used | runtime (s) | GPU Average Utilization | Per GPU FPS |
|---------|---------|------------|------------|-------------|-------------------------|-------------|
| 1       | 206268  | 1920x1080  | 4 / 1*     | 13195.3     | 62.97%                  | 3.91        |
| 2       | 206268  | 1920x1080  | 4 / 1*     | 13125.5     | 61.02%                  | 3.93        |
| 3       | 206268  | 1920x1080  | 4 / 1*     | 13261.9     | 52.63%                  | 3.89        |


## 12/2020, multiple-image batch processing

We test the multiple-image batch processing using these [commands](./README.md#multiple-image-batch-inferencing) for the FPN-ResNet50 model. The machine is with GTX 1070 TI, i7-8700K, SSD. The test video is a single 5-minute video and we test detect and track with 1280x720 resolution, frame_gap=8.

| RunType            | Time | GPU Median Utilization | GPU Average Utilization |
|--------------------|------|------------------------|-------------------------|
|b=1 var             |06:21 |        53.00%          |         54.24%          |
|b=1 frozen          |05:06 |        34.50%          |         36.30%          |
|b=1 frozen,partial  |03:43 |        57.00%          |         49.55%          |
|b=4 var             |04:35 |        50.00%          |         52.48%          |
|b=4 frozen          |03:18 |        64.00%          |         63.32%          |
|b=4 frozen,partial  |03:15 |        42.00%          |         48.05%          |
|b=8 var             |04:27 |        00.00%          |          2.35%          |
|b=8 frozen          |03:12 |        62.00%          |         53.37%          |
|b=8 frozen,partial  |03:07 |        75.50%          |         62.11%          |
