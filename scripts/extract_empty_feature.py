import torch
import os
import numpy as np
import sys
cwd = os.getcwd()
sys.path.append(cwd)
import libs.autoencoder
import libs.clip
from datasets import MSCOCODatabase
import argparse
from tqdm import tqdm



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_emb', default='w2v')
    args = parser.parse_args()
    print(args)
    device = 'cuda'
    save_dir = f'assets/datasets/coco256_features'
    os.makedirs(save_dir, exist_ok=True)
    # empty caption feature
    prompts = ['',]

    if args.text_emb == "clip":       
        clip = libs.clip.FrozenCLIPEmbedder()
        clip.eval()
        clip.to(device)
        latent = clip.encode(prompts)
    elif args.text_emb == "w2v":
        latent = torch.zeros(1,77,768)
  
    c = latent[0].detach().cpu().numpy()
    np.save(os.path.join(save_dir, f'empty_context.npy'), c)



if __name__ == '__main__':
    main()
