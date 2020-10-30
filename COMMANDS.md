# Example Commands

## 02-2020 0.49 pAUDC, 0.64 processing time
```
$ python obj_detect_tracking.py \
 --model_path obj_coco_resnet50_partial_tfv1.14_1280x720_rpn300.pb \
 --video_dir videos --tracking_dir output/ --video_lst_file videos.lst \
 --version 2 --is_coco_model --use_partial_classes  --frame_gap 8 \
 --is_load_from_pb --get_tracking \
 --tracking_objs Person,Vehicle --min_confidence 0.85 \
 --resnet50 --rpn_test_post_nms_topk 300 --max_size 1280 --short_edge_size 720 \
 --use_lijun_video_loader --nms_max_overlap 0.85 --max_iou_distance 0.5 \
 --max_cosine_distance 0.5 --nn_budget 5
```
This is for processing AVI videos. For MP4 videos, run without `--use_lijun`.
Add `--log_time_and_gpu` to get GPU utilization and time profile.


## 05-2020, added EfficientDet
The [EfficientDet (CVPR 2020)](https://github.com/google/automl/tree/master/efficientdet) (D7) is reported to be more than 12 mAP better than the Resnet-50 FPN model we used on COCO.

I have made the following changes based on the code from early May:
+ Added multi-level ROI align with the final detection boxes since we need the FPN box features for deep-SORT tracking. Basically since one-stage object detection models have box predictions at each feature level, I added a level index variable to keep track of each box's feature level so that in the end they can be efficiently backtracked to the original feature map and crop the features.
+ Similar to the MaskRCNN model, I modified the EfficientDet to allow NMS on only some of the COCO classes (currently we only care about person and vehicle) and save computations.


Example command \[[d0 model from early May](https://aladdin-eax.inf.cs.cmu.edu/shares/diva_obj_detect_models/models/efficientdet-d0.tar.gz)\]:
```
$ python obj_detect_tracking.py \
 --model_path efficientdet-d0 \
 --efficientdet_modelname efficientdet-d0 --is_efficientdet \
 --efficientdet_max_detection_topk 5000 \
 --video_dir videos --tracking_dir output/ --video_lst_file videos.lst \
 --version 2 --is_coco_model --use_partial_classes  --frame_gap 8 \
 --get_tracking --tracking_objs Person,Vehicle --min_confidence 0.6 \
 --max_size 1280 --short_edge_size 720 \
 --use_lijun_video_loader --nms_max_overlap 0.85 --max_iou_distance 0.5 \
 --max_cosine_distance 0.5 --nn_budget 5
```
This is for processing AVI videos. For MP4 videos, run without `--use_lijun`.
Add `--log_time_and_gpu` to get GPU utilization and time profile.

Example command with a partial frozen graph \[[d0-TFv1.15](https://aladdin-eax.inf.cs.cmu.edu/shares/diva_obj_detect_models/models/efficientd0_tfv1.15_1280x720.pb)\] (slightly faster):
```
$ python obj_detect_tracking.py \
 --model_path efficientd0_tfv1.15_1280x720.pb --is_load_from_pb \
 --efficientdet_modelname efficientdet-d0 --is_efficientdet \
 --efficientdet_max_detection_topk 5000 \
 --video_dir videos --tracking_dir output/ --video_lst_file videos.lst \
 --version 2 --is_coco_model --use_partial_classes  --frame_gap 8 \
 --get_tracking --tracking_objs Person,Vehicle --min_confidence 0.6 \
 --max_size 1280 --short_edge_size 720 \
 --use_lijun_video_loader --nms_max_overlap 0.85 --max_iou_distance 0.5 \
 --max_cosine_distance 0.5 --nn_budget 5
```

[05/04/2020] Tried to optimize the frozen model with TensorRT by:
```
$ python tensorrt_optimize_tf1.15.py efficientd0_tfv1.15_1280x720.pb \
efficientd0_tfv1.15_1280x720_trt_fp16.pb --precision_mode FP16
```
But it does not work:
```
2020-05-04 22:11:48.850233: F tensorflow/core/framework/op_kernel.cc:875] Check failed: mutable_output(index) == nullptr (0x7f82d4244ff0 vs. nullptr)
Aborted (core dumped)
```

Run object detection and visualization on images. This could be used to reproduce the official repo's tutorial output:
```
$ python obj_detect_imgs.py --model_path efficientdet-d0 \
--efficientdet_modelname efficientdet-d0 --is_efficientdet \
--img_lst imgs.lst --out_dir test_d0_json \
--visualize --vis_path test_d0_vis --vis_thres 0.4 \
--max_size 1920 --short_edge_size 1080 \
--efficientdet_max_detection_topk 5000
```

## 10-2020, comparing EfficientDet with MaskRCNN on video datasets

1. VIRAT

<table>
  <tr>
    <td>Models</td>
    <td>COCO-validation-AP-80classes</td>
    <td>VIRAT Person-Val-AP </td>
    <td>VIRAT Vehicle-Val-AP  </td>
    <td>VIRAT Bike-Val-AP</td>
  </tr>
  <tr>
    <td>MaskRCNN, R50-FPN</td>
    <td>0.389 </td>
    <td>0.374 </td>
    <td>0.943 </td>
    <td>0.367</td>
  </tr>
  <tr>
    <td>MaskRCNN, R101-FPN</td>
    <td>0.407 </td>
    <td>0.378 </td>
    <td>0.947 </td>
    <td>0.399</td>
  </tr>
  <tr>
    <td>EfficientDet-d2</td>
    <td>0.425 </td>
    <td>0.371 </td>
    <td>0.949 </td>
    <td>0.293</td>
  </tr>
  <tr>
    <td>EfficientDet-d6</td>
    <td>0.513 </td>
    <td>0.422 </td>
    <td>0.947 </td>
    <td>0.355</td>
  </tr>
</table>

2. AVA-Kinetics

<table>
  <tr>
    <td>Models</td>
    <td>COCO-validation-AP-80classes</td>
    <td>AVA-Kinetics Train-Person-AP  </td>
    <td>AVA-Kinetics Val-Person-AP</td>
  </tr>
  <tr>
    <td>MaskRCNN, R101-FPN</td>
    <td>0.407 </td>
    <td>0.664 </td>
    <td>0.682</td>
  </tr>
  <tr>
    <td>EfficientDet-d2</td>
    <td>0.425 </td>
    <td>0.650  </td>
    <td>0.680</td>
  </tr>
  <tr>
    <td>EfficientDet-d6</td>
    <td>0.513 </td>
    <td>0.623 </td>
    <td>0.658 </td>
  </tr>
</table>

VIRAT consists of mostly small person boxes, while AVA-Kineitcs has much bigger ones.
