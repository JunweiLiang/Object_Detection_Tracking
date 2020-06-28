# CMU Object Detection & Tracking for Surveillance Video Activity Detection

## Training
- Download the videos from [ActEV](https://actev.nist.gov/) or the dataset you wish to train on and extract all the frames into the following format: `training_frames/${videoname}/${videoname}_F_%08d.jpg`.

- Put annotations into a single folder. one npz file for one frame: `training_annotations/${videoname}_F_%08d.npz`. The filename must match the frame name. You can download my processed annotations and check the data format. Only `labels` and `boxes` are used during training:
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

- Train the obj_v3 model with 1 GPU:
```
$ python main.py nothing training_frames --mode train --annopath v1-training_012019_actgt_allsingle_npz \
--trainlst v1-training_minusminival_frames.lst --train_skip 30 --valframepath v1-training_frames --vallst \
v1-training_minival_frames.lst --valannopath v1-training_012019_actgt_allsingle_npz --outbasepath my_model \
--modelname obj_v3 --num_epochs 15 --save_period 5000 --rpn_batch_size 256 --frcnn_batch_size 512 --num_class \
15 --diva_class3 --max_size 1920 --short_edge_size 1080 --init_lr 0.006 --use_cosine_schedule --warm_up_steps \
10000 --optimizer momentum --rpn_test_post_nms_topk 1000 --freeze 0 --gpu 1 --is_fpn --im_batch_size 1 \
--flip_image --load_from COCO-MaskRCNN-R101FPN1x.npz --skip_first_eval --best_first -1 --show_loss_period 1000 \
--loss_me_step 50 --ignore_vars fastrcnn/outputs --wd 0.0001 --use_dilation --use_frcnn_class_agnostic
```
You can change `--gpu 4` and `--im_batch_size 4` (and maybe `--gpuid_start`) if you have a multi-GPU machine. Note that it is a [known bug](https://github.com/tensorflow/tensorflow/issues/23458) in tf 1.13 that you would see all 4 gpu memory allocated even if you set gpu to 2. This is fixed in tf 1.14.0 (but still takes some GPU0's memory). But multi-GPU training with a subset of the GPUs (`--gpuid_start` larger than 0) will fail since tf v1.13 according to [this](https://github.com/tensorflow/tensorflow/issues/27259).

- June 2020, finetune MaskRCNN person detection on AVA-Kinetics Dataset:
```
$ python main.py nothing pack_ava_kinetics_keyframes --mode train --annopath ava_kinetics_person_box_anno/ --trainlst person_train.lst --valframepath pack_ava_kinetics_keyframes --vallst person_val.lst --valannopath ava_kinetics_person_box_anno/ --outbasepath maskrcnn_finetune --modelname maskrcnn_r101fpn --num_epochs 15 --save_period 5000 --rpn_batch_size 256 --frcnn_batch_size 512 --num_class 81 --is_coco_model --one_level_framepath --max_size 560 --short_edge_size 320 --init_lr 0.001 --use_cosine_schedule --warm_up_steps 10000 --optimizer momentum --rpn_test_post_nms_topk 1000 --freeze 0 --gpu 4 --is_fpn --im_batch_size 4 --flip_image --load_from COCO-MaskRCNN-R101FPN1x.npz --show_loss_period 1000 --loss_me_step 100 --wd 0.0001 --val_skip 10
```

## Testing
- Download the videos from [ActEV](https://actev.nist.gov/) or the dataset you wish to test on and extract all the frames into the following format: `validation_frames/${videoname}/${videoname}_F_%08d.jpg`.

- Put annotations into a single folder. one npz file for one frame: `testing_annotations/${videoname}_F_%08d.npz`. The filename must match the frame name. You can download my processed annotations and check the data format. Only `labels` and `boxes` are used during training:
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
$ python main.py nothing v1-validate_frames.lst --mode forward --outbasepath test_jsons --rpn_batch_size 256 \
--frcnn_batch_size 512 --num_class 15 --diva_class3 --max_size 1920 --short_edge_size 1080 \
--rpn_test_post_nms_topk 1000 --gpu 1 --is_fpn --im_batch_size 1 --load_from my_model/obj_v3/01/save-best/ \
--use_frcnn_class_agnostic --use_dilation
```

- Evaluate:
```
$ python eval.py v1-validate_frames.lst v1-validate_012019_actgt_allsingle_npz test_jsons/
```

