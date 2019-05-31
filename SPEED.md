# Experiments on Inferencing Speed



## Experiment 1: All runs get almost exactly the same outputs

Machine 1: 2 GTX 1080 TI, i7, nvme

Machine 2: 3 GTX 1080 TI + 1 TITAN X, E5, nvme

Each run is conducted without other programs running.

Machine 1

| Run #1  |         |            |            |             |                         |             |
|---------|---------|------------|------------|-------------|-------------------------|-------------|
| RunType | # Image | Image Size | # GPU Used | runtime (s) | GPU Average Utilization | Per GPU FPS |
| 1       | 206268  | 1920x1080  | 2          | 41190.9     | 65.3%                   | 2.50        |
| 2       | 206268  | 1920x1080  | 2          | 37920.9     | 62.0%                   | 2.72        |
| 3       | 206268  | 1920x1080  | 2          | 31494.2     | 54.8%                   | 3.27        |
| 2       | 206268  | 1920x1080  | 1          |             |                         |             |
| 3       | 206268  | 1920x1080  | 1          |             |                         |             |
| Run #2  |         |            |            |             |                         |             |
| RunType | # Image | Image Size | # GPU Used | runtime (s) | GPU Average Utilization | Per GPU FPS |
| 2       | 206268  | 1920x1080  | 2          |             |                         |             |
| 3       | 206268  | 1920x1080  | 2          |             |                         |             |
| 2       | 206268  | 1920x1080  | 1          |             |                         |             |
| 3       | 206268  | 1920x1080  | 1          |             |                         |             |

Machine 2

| Run #1  |         |            |            |             |                         |             |
|---------|---------|------------|------------|-------------|-------------------------|-------------|
| Summary | # Image | Image Size | # GPU Used | runtime (s) | GPU Average Utilization | Per GPU FPS |
| 2       | 206268  | 1920x1080  | 4          | 29354.4     | 33.8%                   | 1.76        |
| 3       | 206268  | 1920x1080  | 4          | 26454.0     | 23.5%                   | 1.95        |
| 2       | 206268  | 1920x1080  | 2          |             |                         |             |
| 3       | 206268  | 1920x1080  | 2          |             |                         |             |
| 2       | 206268  | 1920x1080  | 1          |             |                         |             |
| 3       | 206268  | 1920x1080  | 1          |             |                         |             |
| Run #2  |         |            |            |             |                         |             |
| Summary | # Image | Image Size | # GPU Used | runtime (s) | GPU Average Utilization | Per GPU FPS |
| 2       | 206268  | 1920x1080  | 4          |             |                         |             |
| 3       | 206268  | 1920x1080  | 4          |             |                         |             |
| 2       | 206268  | 1920x1080  | 2          |             |                         |             |
| 3       | 206268  | 1920x1080  | 2          |             |                         |             |
| 2       | 206268  | 1920x1080  | 1          |             |                         |             |
| 3       | 206268  | 1920x1080  | 1          |             |                         |             |
| 3       | 206268  | 1920x1080  | 4 / 1*     |             |                         |             |

TODO: Add input queues mechanism to improve GPU utilization.

| RunType |                                                                         |
|---------|-------------------------------------------------------------------------|
| 1       | tf 1.10 (CUDA 9, cudnn 7.1), Variable Model                             |
| 2       | tf 1.13 (CUDA 10.0 cudnn 7.4), Variable Model                           |
| 3       | tf 1.13 (CUDA 10.0 cudnn 7.4), Frozen Graph (.pb)                       |
| 4       | tf 1.13 (CUDA 10.0 cudnn 7.4), Frozen Graph (.pb) -> TensorRT Optimized |

To freeze the model into a .pb file:
```
$ python main.py nothing nothing --mode pack --pack_model_path obj_v3.pb \
--pack_modelconfig_path obj_v3.config.json --load_from obj_v3_model/ --note obj_v3 --num_class 15 \
--diva_class3 --rpn_batch_size 256 --frcnn_batch_size 512 --rpn_test_post_nms_topk 1000 --is_fpn \
--use_dilation --max_size 1920 --short_edge_size 1080
```

Run testing on the v1-val set:
```
$ python main.py nothing v1-validate_frames.lst --mode forward --outbasepath obj_v3_val_output --num_class 15 --diva_class3 --max_size 1920 --short_edge_size 1080 --gpu 2 --im_batch_size 2 --load_from obj_v3.pb  --is_load_from_pb --log
```
Assuming `v1-validate_frames.lst` contains absolute path of all images. This will output one json in COCO detection format for each image in `obj_v3_val_output/`. The `--log` will run `nvidia-smi` every couple second in a separate thread to record the gpu utilizations.

## Experiment 2: TensorRT Optimization (Not Working Yet)

Given the frozen .pb file, I use the following to get TensorRT optimized graph (I install dependencies according to [this](https://www.tensorflow.org/install/gpu#ubuntu_1604_cuda_10)):
```
$ python tensorrt_optimize.py obj_v3.pb obj_v3_tensorrt.pb
```

But when I run testing with:
```
$ python main.py nothing v1-validate_frames.lst --mode forward --outbasepath obj_v3_val_output --num_class 15 --diva_class3 --max_size 1920 --short_edge_size 1080 --gpu 2 --im_batch_size 2 --load_from obj_v3_tensorrt.pb  --is_load_from_pb --log
```
I got the following error:
`InternalError (see above for traceback): Native FunctionDef TRTEngineOp_34_native_segment can't be found in function library`