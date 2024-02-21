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
* MS-COCO: Download COCO 2014 [training](http://images.cocodataset.org/zips/train2014.zip), [validation](http://images.cocodataset.org/zips/val2014.zip) data and [annotations](http://images.cocodataset.org/annotations/annotations_trainval2014.zip). 
```sh
python scripts/extract_mscoco_feature.py
python scripts/extract_mscoco_feature.py --split=val
python scripts/extract_test_prompt_feature.py
python scripts/extract_empty_feature.py
```
#### Reference statistics for FID
Download `fid_stats` directory from this [link](https://drive.google.com/drive/folders/1yo-XhqbPue3rp5P57j6QbA5QZx6KybvP?usp=sharing) (which contains reference statistics for FID).
Put the downloaded directory as `assets/fid_stats` in this codebase.
In addition to evaluation, these reference statistics are used to monitor FID during training.
## Configs
In config files
```sh
config.nnet = d(
    name='uvit_t2i'
    ...,
    c = c,
    v = v,
    ...
)
# change c and v for caption and image transformer depths
# change name to 'uvit_t2i_old','uvit_t2i_cross','uvit_t2i', for original U-ViT-small, cross-attention, and self-attention Models
# name='uvit_t2i', c=0, v=0 is equivalent to U-ViT-small, but cannot load the pretrained weights provided by U-ViT paper.
# name='uvit_t2i_old' will ignore c and v values.
```
## Training
We use the [huggingface accelerate](https://github.com/huggingface/accelerate) library to help train with distributed data parallel and mixed precision. The following is the training command:

```sh
# MS-COCO (U-ViT-S/2)
accelerate launch --num_processes 1 --mixed_precision fp16 train_t2i_discrete.py --config=configs/mscoco_uvit_small.py
```

## Sampling
```sh
# Running will store the images generated from prompt file test.txt at --nnet_path
accelerate launch --num_processes 1 --mixed_precision fp16 sample_t2i_discrete.py --config=configs/mscoco_uvit_small.py --nnet_path=nnet.pth --input_path=test.txt
```

## Evaluation (MS-COCO (U-ViT-S/2))

```sh
# FID
accelerate launch --multi_gpu --num_processes 1 --mixed_precision fp16 eval_t2i_discrete.py --config=configs/mscoco_uvit_small.py --nnet_path=nnet.pth

# CLIP Score
# The first JSON file containing 30000 test captions will be extracted by running 'python scripts/extract_mscoco_feature.py --split=val'
python tools/clipscore.py assets/datasets/coco256_features/val/eval_captions/captions.json workdir/*/*/ckpts/*.ckpt/eval_samples/
```

## References


This implementation is based on
* [Extended Analytic-DPM](https://github.com/baofff/Extended-Analytic-DPM) (provide the FID reference statistics on CIFAR10 and CelebA 64x64)
* [guided-diffusion](https://github.com/openai/guided-diffusion) (provide the FID reference statistics on ImageNet)
* [pytorch-fid](https://github.com/mseitzer/pytorch-fid) (provide the official implementation of FID to PyTorch)
* [dpm-solver](https://github.com/LuChengTHU/dpm-solver) (provide the sampler)
