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


## 04-2020, added EfficientDet
The [EfficientDet (CVPR 2020)](https://github.com/google/automl/tree/master/efficientdet) is reported to be more than 14 mAP better than the Resnet-50 FPN model we used on COCO.

I have made the following changes based on the code from early March:
+ The original code assumes width==height and it will pad (1280x720) frame to (1280x1280) at the beginning, which wastes much computation. See [this issue](https://github.com/google/automl/issues/162). This is an easy fix. Note that I make sure the image sizes are multipliers of 128 (2^7) with some paddings. So (1280x720) inputs would be (1280x768).
+ Added multi-level ROI align with the final detection boxes since we need the FPN box features for deep-SORT tracking. Basically since one-stage object detection models have box predictions at each feature level, I added a level index variable to keep track of each box's feature level so that in the end they can be efficiently backtracked to the original feature map and crop the features.
+ Similar to the MaskRCNN model, I modified the EfficientDet to allow NMS on only some of the COCO classes (currently we only care about person and vehicle) and save computations.
+ Separate the tf.py_func stuff since this part of the graph cannot be saved to a .pb model. (The official EfficientDet code is still actively being developed and this problem seems to have been solved. Will look into this later.)

Example command \[[d0 model from early March](https://aladdin-eax.inf.cs.cmu.edu/shares/diva_obj_detect_models/models/efficientdet-d0.tar.gz)\]:
```
$ python obj_detect_tracking.py \
 --model_path efficientdet-d0 \
 --efficientdet_modelname efficientdet-d0 --is_efficientdet \
 --efficientdet_max_detection_topk 1000 \
 --video_dir videos --tracking_dir output/ --video_lst_file videos.lst \
 --version 2 --is_coco_model --use_partial_classes  --frame_gap 8 \
 --get_tracking --tracking_objs Person,Vehicle --min_confidence 0.6 \
 --max_size 1280 --short_edge_size 720 \
 --use_lijun_video_loader --nms_max_overlap 0.85 --max_iou_distance 0.5 \
 --max_cosine_distance 0.5 --nn_budget 5
```
This is for processing AVI videos. For MP4 videos, run without `--use_lijun`.
Add `--log_time_and_gpu` to get GPU utilization and time profile.

Example command with a partial frozen graph \[[d0-TFv1.15](https://aladdin-eax.inf.cs.cmu.edu/shares/diva_obj_detect_models/models/efficientd0_tfv1.15_1280x720.pb)\] (this is not faster than above):
```
$ python obj_detect_tracking.py \
 --model_path efficientd0_tfv1.15_1280x720.pb --is_load_from_pb \
 --efficientdet_modelname efficientdet-d0 --is_efficientdet \
 --efficientdet_max_detection_topk 1000 \
 --video_dir videos --tracking_dir output/ --video_lst_file videos.lst \
 --version 2 --is_coco_model --use_partial_classes  --frame_gap 8 \
 --get_tracking --tracking_objs Person,Vehicle --min_confidence 0.6 \
 --max_size 1280 --short_edge_size 720 \
 --use_lijun_video_loader --nms_max_overlap 0.85 --max_iou_distance 0.5 \
 --max_cosine_distance 0.5 --nn_budget 5
```
