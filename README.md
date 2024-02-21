## U-ViT<br> <sub><small>Official PyTorch implementation of xxxx</small></sub>
--------------------

## Dependency

```sh
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116  # install torch-1.13.1
pip install accelerate==0.12.0 absl-py ml_collections einops wand ftfy==6.1.1 transformers==4.23.1

pip install -U xformers
pip install -U --pre triton
```

## Preparation Before Training and Evaluation

#### Autoencoder
Download `stable-diffusion` directory from this [link](https://drive.google.com/drive/folders/1yo-XhqbPue3rp5P57j6QbA5QZx6KybvP?usp=sharing) (which contains image autoencoders converted from [Stable Diffusion](https://github.com/CompVis/stable-diffusion)). 
Put the downloaded directory as `assets/stable-diffusion` in this codebase.
The autoencoders are used in latent diffusion models.

#### Data
* MS-COCO: Download COCO 2014 [training](http://images.cocodataset.org/zips/train2014.zip), [validation](http://images.cocodataset.org/zips/val2014.zip) data and [annotations](http://images.cocodataset.org/annotations/annotations_trainval2014.zip). Then extract their features according to `scripts/extract_mscoco_feature.py` `scripts/extract_test_prompt_feature.py` `scripts/extract_empty_feature.py`.

#### Reference statistics for FID
Download `fid_stats` directory from this [link](https://drive.google.com/drive/folders/1yo-XhqbPue3rp5P57j6QbA5QZx6KybvP?usp=sharing) (which contains reference statistics for FID).
Put the downloaded directory as `assets/fid_stats` in this codebase.
In addition to evaluation, these reference statistics are used to monitor FID during training.

## Training
We use the [huggingface accelerate](https://github.com/huggingface/accelerate) library to help train with distributed data parallel and mixed precision. The following is the training command:

```sh
# MS-COCO (U-ViT-S/2)
accelerate launch --num_processes 1 --mixed_precision fp16 train_t2i_discrete.py --config=configs/mscoco_uvit_small.py

## Evaluation (Compute FID)

We use the [huggingface accelerate](https://github.com/huggingface/accelerate) library for efficient inference with mixed precision and multiple gpus. The following is the evaluation command:
```sh
# the evaluation setting
num_processes=2  # the number of gpus you have, e.g., 2
eval_script=eval.py  # the evaluation script, one of <eval.py|eval_ldm.py|eval_ldm_discrete.py|eval_t2i_discrete.py>
                     # eval.py: for models trained with train.py (i.e., pixel space models)
                     # eval_ldm.py: for models trained with train_ldm.py (i.e., latent space models with continuous timesteps)
                     # eval_ldm_discrete.py: for models trained with train_ldm_discrete.py (i.e., latent space models with discrete timesteps)
                     # eval_t2i_discrete.py: for models trained with train_t2i_discrete.py (i.e., text-to-image models on latent space)
config=configs/cifar10_uvit_small.py  # the training configuration

# launch evaluation
accelerate launch --multi_gpu --num_processes $num_processes --mixed_precision fp16 eval_script --config=$config
```
The generated images are stored in a temperary directory, and will be deleted after evaluation. If you want to keep these images, set `--config.sample.path=/save/dir`.


We provide all commands to reproduce FID results in the paper:
```sh
# CIFAR10 (U-ViT-S/2)
accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 eval.py --config=configs/cifar10_uvit_small.py --nnet_path=cifar10_uvit_small.pth

# CelebA 64x64 (U-ViT-S/4)
accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 eval.py --config=configs/celeba64_uvit_small.py --nnet_path=celeba64_uvit_small.pth

# ImageNet 64x64 (U-ViT-M/4)
accelerate launch --multi_gpu --num_processes 8 --mixed_precision fp16 eval.py --config=configs/imagenet64_uvit_mid.py --nnet_path=imagenet64_uvit_mid.pth

# ImageNet 64x64 (U-ViT-L/4)
accelerate launch --multi_gpu --num_processes 8 --mixed_precision fp16 eval.py --config=configs/imagenet64_uvit_large.py --nnet_path=imagenet64_uvit_large.pth

# ImageNet 256x256 (U-ViT-L/2)
accelerate launch --multi_gpu --num_processes 8 --mixed_precision fp16 eval_ldm.py --config=configs/imagenet256_uvit_large.py --nnet_path=imagenet256_uvit_large.pth

# ImageNet 256x256 (U-ViT-H/2)
accelerate launch --multi_gpu --num_processes 8 --mixed_precision fp16 eval_ldm_discrete.py --config=configs/imagenet256_uvit_huge.py --nnet_path=imagenet256_uvit_huge.pth

# ImageNet 512x512 (U-ViT-L/4)
accelerate launch --multi_gpu --num_processes 8 --mixed_precision fp16 eval_ldm.py --config=configs/imagenet512_uvit_large.py --nnet_path=imagenet512_uvit_large.pth

# ImageNet 512x512 (U-ViT-H/4)
accelerate launch --multi_gpu --num_processes 8 --mixed_precision fp16 eval_ldm_discrete.py --config=configs/imagenet512_uvit_huge.py --nnet_path=imagenet512_uvit_huge.pth

# MS-COCO (U-ViT-S/2)
accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 eval_t2i_discrete.py --config=configs/mscoco_uvit_small.py --nnet_path=mscoco_uvit_small.pth

# MS-COCO (U-ViT-S/2, Deep)
accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 eval_t2i_discrete.py --config=configs/mscoco_uvit_small.py --config.nnet.depth=16 --nnet_path=mscoco_uvit_small_deep.pth
```




## References
If you find the code useful for your research, please consider citing
```bib
@inproceedings{bao2022all,
  title={All are Worth Words: A ViT Backbone for Diffusion Models},
  author={Bao, Fan and Nie, Shen and Xue, Kaiwen and Cao, Yue and Li, Chongxuan and Su, Hang and Zhu, Jun},
  booktitle = {CVPR},
  year={2023}
}
```

This implementation is based on
* [Extended Analytic-DPM](https://github.com/baofff/Extended-Analytic-DPM) (provide the FID reference statistics on CIFAR10 and CelebA 64x64)
* [guided-diffusion](https://github.com/openai/guided-diffusion) (provide the FID reference statistics on ImageNet)
* [pytorch-fid](https://github.com/mseitzer/pytorch-fid) (provide the official implementation of FID to PyTorch)
* [dpm-solver](https://github.com/LuChengTHU/dpm-solver) (provide the sampler)
