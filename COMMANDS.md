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
