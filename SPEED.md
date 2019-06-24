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

## Experiment 2: TensorRT Optimization (Not Working Yet)

Given the frozen .pb file, I use the following to get TensorRT optimized graph (I install dependencies according to [this](https://www.tensorflow.org/install/gpu#ubuntu_1604_cuda_10)):
```
$ python tensorrt_optimize.py obj_v3.pb obj_v3_tensorrt.pb
```

But when I run testing with:
```
$ python main.py nothing v1-validate_frames.lst --mode forward --outbasepath \
obj_v3_val_output --num_class 15 --diva_class3 --max_size 1920 --short_edge_size \
1080 --gpu 2 --im_batch_size 2 --load_from obj_v3_tensorrt.pb  --is_load_from_pb --log
```
I got the following error:
`InternalError (see above for traceback): Native FunctionDef TRTEngineOp_34_native_segment can't be found in function library`

Changed `is_dynamic_op=False` and then I got this error:
```
InternalError (see above for traceback): Native FunctionDef fpn/upsample_lat5/Tensordot/TRTEngineOp_39_native_segment can't be found in function library
         [[node model_0/fpn/upsample_lat5/Tensordot/TRTEngineOp_39 (defined at /home/junweil/object_detection/script/tf_mrcnn/models.py:147) ]]
         [[{{node model_0/fastrcnn_predictions/map/while/body/_1/GatherV2_1}}]]
```

These errors are likely caused by my use of [unsupported OPs](https://github.com/tensorflow/tensorrt/issues/80) in TensorRT.
