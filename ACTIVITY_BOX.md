# Frame-level Activity Detection Experiments

## Training
- Download the videos from [ActEV](https://actev.nist.gov/) or the dataset you wish to train on and extract all the frames into the following format: `training_frames/${videoname}/${videoname}_F_%08d.jpg`.

- Put annotations into a single folder. one npz file for one frame: `training_annotations/${videoname}_F_%08d.npz`. The filename must match the frame name. You can download my processed annotations and check the data format. Here `actlabels` and `actboxes` are used during training:
```
wget https://aladdin-eax.inf.cs.cmu.edu/shares/diva_obj_detect_models/v1-training_012019_actgt_allsingle_npz.tar
```

- Prepare the file list for training set and validation set. We split a small subset of the ActEV training set as the validation set and the ActEV validation set will be used for testing. You can download my file lst. Training set:
```
wget https://aladdin-eax.inf.cs.cmu.edu/shares/diva_obj_detect_models/v1-training_minusminival_frames.lst
```
Validation set:
```
wget https://aladdin-eax.inf.cs.cmu.edu/shares/diva_obj_detect_models/v1-training_minival_frames.lst
```
These file lists are in absolute path. You will need to replace the path with the correct ones.

- Download MSCOCO pretrained model from Tensorpack:
```
wget http://models.tensorpack.com/FasterRCNN/COCO-MaskRCNN-R101FPN1x.npz
```

- Train the Actbox v1 model with 1 GPU:
```
$python main.py nothing training_frames --mode train --annopath v1-training_012019_actgt_allsingle_npz \
--trainlst v1-training_minusminival_frames.lst --train_skip 5 --valframepath \
v1-training_frames --vallst v1-training_minival_frames.lst --valannopath \
v1-training_012019_actgt_allsingle_npz --outbasepath bupt_actboxexp_resnet101_dilation_classagnostic --modelname mrcnn101 --num_epochs 20 \
--save_period 2500 --rpn_batch_size 256 --frcnn_batch_size 512 --num_class 10 \
--bupt_exp --max_size 1920 --short_edge_size 1080 --init_lr 0.003 --use_cosine_schedule \
--warm_up_steps 5000 --optimizer momentum --rpn_test_post_nms_topk 1000 --freeze 0 \
--gpu 1 --is_fpn --im_batch_size 1 --flip_image --val_skip 20 --load_from \
COCO-R101FPN-MaskRCNN-Standard.npz --skip_first_eval --best_first -1 --show_loss_period \
1000 --loss_me_step 50 --ignore_vars fastrcnn/outputs --wd 0.0001 --use_dilation \
--use_frcnn_class_agnostic
```
You can change `--gpu 4` and `--im_batch_size 4` (and maybe `--gpuid_start`) if you have a multi-GPU machine. Note that it is a [known bug](https://github.com/tensorflow/tensorflow/issues/23458) in tf 1.13 that you would see all 4 gpu memory allocated even if you set gpu to 2. This is fixed in tf 1.14.0 (but still takes some GPU0's memory). But multi-GPU training with a subset of the GPUs (`--gpuid_start` larger than 0) will fail since tf v1.13 according to [this](https://github.com/tensorflow/tensorflow/issues/27259).

## Testing
- Download the videos from [ActEV](https://actev.nist.gov/) or the dataset you wish to test on and extract all the frames into the following format: `validation_frames/${videoname}/${videoname}_F_%08d.jpg`.

- Put annotations into a single folder. one npz file for one frame: `testing_annotations/${videoname}_F_%08d.npz`. The filename must match the frame name. You can download my processed annotations and check the data format. Here `actlabels` and `actboxes` are used:
```
wget https://aladdin-eax.inf.cs.cmu.edu/shares/diva_obj_detect_models/v1-validate_012019_actgt_allsingle_npz.tar
```

- Prepare the file list for testing. We use the official validation set as testing set. You can download my file lst:
```
wget https://aladdin-eax.inf.cs.cmu.edu/shares/diva_obj_detect_models/v1-validate_frames.lst
```
Again, you will need to replace them with the correct absolute path.

- Test the model by generating COCO-format JSON files:
```
$ python main.py nothing v1-validate_frames.lst --mode forward --outbasepath \
actbox_v1_test --rpn_batch_size 256 --frcnn_batch_size 512 --num_class 10 --bupt_exp \
--max_size 1920 --short_edge_size 1080 --rpn_test_post_nms_topk 1000 --gpu 1 --is_fpn \
--im_batch_size 1 --load_from bupt_actboxexp_resnet101_dilation_classagnostic/ \
mrcnn101/01/save-best/ --use_dilation --use_frcnn_class_agnostic --log
```

- Evaluate:
```
$ python eval.py v1-validate_frames.lst v1-validate_012019_actgt_allsingle_npz \
actbox_v1_test --bupt_exp
```

- Visualize:
```
$ python vis_json.py v1-validate_videos.lst v1-validate_frames/ actbox_v1_test/ \
actbox_v1_test_visbox --score_thres 0.7
```

- Tracking
```
$ python obj_detect_tracking.py --model_path bupt_actboxexp_resnet101_dilation_classagnostic/mrcnn101/01/save-best/ --version 5 \
--video_dir v1-validate_videos/ --video_lst_file v1-validate_videos.names.lst --out_dir \
act_box_out --frame_gap 1 --get_tracking --tracking_dir act_track_out --min_confidence \
0.8 --tracking_objs Person-Vehicle,Pull,Riding,Talking,Transport_HeavyCarry,Vehicle-Turning,activity_carrying \
--bupt_exp --num_class 10 --gpuid_start 0
```
