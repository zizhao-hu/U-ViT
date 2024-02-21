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
import json

def main(resolution=256):
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='train')
    parser.add_argument('--text_emb', default='clip')
    args = parser.parse_args()
    print(args)


    if args.split == "train":
        datas = MSCOCODatabase(root='assets/datasets/coco/train2014',
                             annFile='assets/datasets/coco/annotations/captions_train2014.json',
                             size=resolution)
        save_dir = f'assets/datasets/coco{resolution}_features/train'
    elif args.split == "val":
        datas = MSCOCODatabase(root='assets/datasets/coco/val2014',
                             annFile='assets/datasets/coco/annotations/captions_val2014.json',
                             size=resolution)
        save_dir = f'assets/datasets/coco{resolution}_features/val'
    else:
        raise NotImplementedError("ERROR!")

    device = "cuda"
    # os.makedirs(save_dir)

    # autoencoder = libs.autoencoder.get_model('assets/stable-diffusion/autoencoder_kl.pth')
    # autoencoder.to(device)

    # if args.text_emb == "clip":
    #     clip = libs.clip.FrozenCLIPEmbedder()
    #     clip.eval()
    #     clip.to(device)
    # elif args.text_emb == "w2v":
    #     wv = libs.emb.WV('assets/text_embedder/w2v_coco_768.model')

    with torch.no_grad():
        eval_captions_dir = os.path.join(save_dir, 'eval_captions')
        if not os.path.exists(eval_captions_dir):
            os.makedirs(eval_captions_dir)
        captions_dict = {}

        for idx, data in tqdm(enumerate(datas)):
            x, captions = data
            # if len(x.shape) == 3:
            #     x = x[None, ...]
            # x = torch.tensor(x, device=device)
            # moments = autoencoder(x, fn='encode_moments').squeeze(0)
            # moments = moments.detach().cpu().numpy()
            # np.save(os.path.join(save_dir, f'{idx}.npy'), moments)

            # if args.text_emb == "clip": 
            #     latent = clip.encode(captions)
            # elif args.text_emb == "w2v":
            #     latent = wv.encode(captions)

            # for i in range(len(latent)):
            #     c = latent[i].detach().cpu().numpy()
            #     np.save(os.path.join(save_dir, f'{idx}_{i}.npy'), c)

            if idx < 30000 and args.split == "val":
                captions_dict[str(idx)] = captions[0]
            else: break
    json_file_path = os.path.join(eval_captions_dir, 'captions.json')
    with open(json_file_path, 'w') as json_file:
        json.dump(captions_dict, json_file)


if __name__ == '__main__':
    main()
