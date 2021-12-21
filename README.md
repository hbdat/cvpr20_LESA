# Shared Attention for Multi-label Zero-shot Learning

## Overview
This repository contains the implementation of [Shared Attention for Multi-label Zero-shot Learning](http://khoury.neu.edu/home/eelhami/publications/MultiAtt-MLZSL-CVPR20.pdf).
> In this work, we address zero-shot multi-label learning for recognition all (un)seen labels using a shared multi-attention method with a novel training mechanism.

![Image](https://github.com/hbdat/cvpr20_LESA/raw/master/fig/high_level_schematic.png)

---
## Prerequisites
+ Python 3.x
+ TensorFlow 1.8.0
+ sklearn
+ matplotlib
+ skimage
+ scipy==1.4.1

---
## Data Preparation

Please download and extract the vgg_19 model (http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz) in `./model/vgg_19`. Make sure the extract model is named `vgg_19.ckpt`

### NUS-WIDE

1) Please download NUS-WIDE images and meta-data into `./data/NUS-WIDE` folder according to the instructions within the folders `./data/NUS-WIDE` and `./data/NUS-WIDE/Flickr`.

2) To extract features into TensorFlow storage format, please run:
```
python ./extract_data/extract_full_NUS_WIDE_images_VGG_feature_2_TFRecord.py			#`data_set` == `Train`: create NUS_WIDE_Train_full_feature_ZLIB.tfrecords
python ./extract_data/extract_full_NUS_WIDE_images_VGG_feature_2_TFRecord.py			#`data_set` == `Test`: create NUS_WIDE_Test_full_feature_ZLIB.tfrecords
```
Please change the `data_set` variable in the script to `Train` and `Test` to extract `NUS_WIDE_Train_full_feature_ZLIB.tfrecords` and `NUS_WIDE_Test_full_feature_ZLIB.tfrecords`.

### Open Images

1) Please download Open Images urls and annotation into `./data/OpenImages` folder according to the instructions within the folders `./data/OpenImages/2017_11` and `./data/OpenImages/2018_04`.

2) To crawl images from the web, please run the script:
```
python ./download_imgs/asyn_image_downloader.py 					#`data_set` == `train`: download images into `./image_data/train/`
python ./download_imgs/asyn_image_downloader.py 					#`data_set` == `validation`: download images into `./image_data/validation/`
python ./download_imgs/asyn_image_downloader.py 					#`data_set` == `test`: download images into `./image_data/test/`
```
Please change the `data_set` variable in the script to `train`, `validation`, and `test` to download different data splits.

3) To extract features into TensorFlow storage format, please run:
```
python ./extract_data/extract_images_VGG_feature_2_TFRecord.py						#`data_set` == `train`: create train_feature_2018_04_ZLIB.tfrecords
python ./extract_data/extract_images_VGG_feature_2_TFRecord.py						#`data_set` == `validation`: create validation_feature_2018_04_ZLIB.tfrecords
python ./extract_data/extract_test_seen_unseen_images_VGG_feature_2_TFRecord.py			        #`data_set` == `test`:  create OI_seen_unseen_test_feature_2018_04_ZLIB.tfrecords
```
Please change the `data_set` variable in the `extract_images_VGG_feature_2_TFRecord.py` script to `train`, and `validation` to extract features from different data splits.

---
## Training and Evaluation

### NUS-WIDE

1) To train and evaluate zero-shot learning model on full NUS-WIDE dataset, please run:
```
python ./zeroshot_experiments/NUS_WIDE_zs_rank_Visual_Word_Attention.py
```

### Open Images

1) To train our framework, please run:
```
python ./multilabel_experiments/OpenImage_rank_Visual_Word_Attention.py				#create a model checkpoint in `./results`
```
2) To evaluate zero-shot performance, please run:
```
python ./zeroshot_experiments/OpenImage_evaluate_top_multi_label.py					#set `evaluation_path` to the model checkpoint created in step 1) above
```
Please set the `evaluation_path` variable to the model checkpoint created in step 1) above

---
## Pretrained Model
We also include the checkpoint of the zero-shot model on NUS-WIDE for fast evaluation (`./results/release_zs_NUS_WIDE_log_GPU_7_1587185916d2570488/`)

---
## Citation
If this code is helpful for your research, we would appreciate if you cite the work:
```
@article{Huynh-LESA:CVPR20,
  author = {D.~Huynh and E.~Elhamifar},
  title = {A Shared Multi-Attention Framework for Multi-Label Zero-Shot Learning},
  journal = {{IEEE} Conference on Computer Vision and Pattern Recognition},
  year = {2020}}
```
