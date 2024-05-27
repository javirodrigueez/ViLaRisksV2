# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# arXiv:
# Mengmeng Wang, Jiazheng Xing, Yong Liu

import os
import clip
import torch.nn as nn
from datasets import Action_DATASETS
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import argparse
import shutil
from pathlib import Path
import yaml
from dotmap import DotMap
import pprint
import numpy
from modules_action.Visual_Prompt import visual_prompt
from utils.Augmentation import get_augmentation
import torch
from utils.Text_Prompt import *

class TextCLIP(nn.Module):
    def __init__(self, model):
        super(TextCLIP, self).__init__()
        self.model = model

    def forward(self, text):
        return self.model.encode_text(text)

class ImageCLIP(nn.Module):
    def __init__(self, model):
        super(ImageCLIP, self).__init__()
        self.model = model

    def forward(self, image):
        return self.model.encode_image(image)

def validate(epoch, val_loader, classes, device, model, fusion_model, config, num_text_aug, save_output):
    model.eval()
    fusion_model.eval()
    num = 0
    corr_1 = 0
    corr_5 = 0
    corr_3 = 0
    similarities_cpu = []
    filenames = []
    labels = []

    with torch.no_grad():
        text_inputs = classes.to(device)
        text_features = model.encode_text(text_inputs)
        f = open('saved_tensors/top5.txt', 'w')
        for iii, (image, class_id, filename) in enumerate(tqdm(val_loader)):
            image = image.view((-1, config.data.num_segments, 3) + image.size()[-2:])
            b, t, c, h, w = image.size()
            #class_id = class_id.to(device)
            image_input = image.to(device).view(-1, c, h, w)
            image_features = model.encode_image(image_input).view(b, t, -1)
            image_features = fusion_model(image_features)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T)
            #import pdb; pdb.set_trace()
            similarity = similarity.view(b, num_text_aug, -1).softmax(dim=-1)
            similarity = similarity.mean(dim=1, keepdim=False)
            # Modified
            sim_cpu = similarity.cpu()
            similarities_cpu.append(sim_cpu)
            filenames.extend(filename)
            labels = []
            for i in class_id:
                label = torch.tensor([int(j) for j in i.split(';')])
                labels.append(label)
            print(labels)
            print('#####################')
            # --------------------------------
            values_1, indices_1 = similarity.topk(1, dim=-1)
            values_5, indices_5 = similarity.topk(5, dim=-1)
            values_3, indices_3 = similarity.topk(3, dim=-1)
            indices_1 = indices_1.cpu()
            indices_5 = indices_5.cpu()
            indices_3 = indices_3.cpu()
            print(indices_3)
            num += b
            if save_output != '':
                import csv
                f_risks = open(save_output, 'a')
                question = 'What is doing the person?'
                writer = csv.writer(f_risks)
                indices_5_str = [str(i.item()) for i in indices_5[0]]
                indices_5_str = ';'.join(indices_5_str)
                writer.writerow([question, indices_5_str, 'none'])
                f_risks.close()
            for i in range(b):
                if indices_1[i] in labels[i]:
                    corr_1 += 1
                for j in indices_5[i]:
                    if j in labels[i]:
                        corr_5 += 1
                        break
                for j in indices_3[i]:
                    if j in labels[i]:
                        corr_3 += 1
                        break
                indices_5_str = [str(i.item()) for i in indices_5[i]]
                f.write(f'{filename[i]} {class_id[i]} {';'.join(indices_5_str)}\n')
                # if class_id[i] in indices_5[i]:
                #     corr_5 += 1
        f.close()
    similarities_cpu = torch.cat(similarities_cpu, dim=0)
    torch.save(similarities_cpu, os.path.join('./saved_tensors', 'similarities_cpu.pt'))
    with open(os.path.join('./saved_tensors', 'filenames.txt'), 'w') as f:
        for file in filenames:
            f.write(f'{file}\n')
    top1 = float(corr_1) / num * 100
    top5 = float(corr_5) / num * 100
    top3 = float(corr_3) / num * 100
    wandb.log({"top1": top1})
    wandb.log({"top5": top5})
    print('Epoch: [{}/{}]: Top1: {}, Top3: {}, Top5: {}'.format(epoch, config.solver.epochs, top1, top3, top5))
    return top1

def main():
    global args, best_prec1
    global global_step
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', default='')
    parser.add_argument('--log_time', default='')
    parser.add_argument('--save_output', default='')
    args = parser.parse_args()

    f = open(args.config, 'r')
    config = yaml.load(f, Loader=yaml.FullLoader)
    f.close()
    working_dir = os.path.join('./exp', config['network']['type'], config['network']['arch'], config['data']['dataset'],
                               args.log_time)
    wandb.init(mode='disabled')
    # wandb.init(project=config['network']['type'],
    #            name='{}_{}_{}_{}'.format(args.log_time, config['network']['type'], config['network']['arch'],
    #                                      config['data']['dataset']))
    print('-' * 80)
    print(' ' * 20, "working dir: {}".format(working_dir))
    print('-' * 80)

    print('-' * 80)
    print(' ' * 30, "Config")
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(config)
    print('-' * 80)

    config = DotMap(config)

    Path(working_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, working_dir)
    shutil.copy('test.py', working_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.

    model, clip_state_dict = clip.load(config.network.arch, device=device, jit=False, tsm=config.network.tsm,
                                                   T=config.data.num_segments, dropout=config.network.drop_out,
                                                   emb_dropout=config.network.emb_dropout)  # Must set jit=False for training  ViT-B/32

    transform_val = get_augmentation(False, config)

    fusion_model = visual_prompt(config.network.sim_header, clip_state_dict, config.data.num_segments)

    model_text = TextCLIP(model)
    model_image = ImageCLIP(model)

    model_text = torch.nn.DataParallel(model_text).cuda()
    model_image = torch.nn.DataParallel(model_image).cuda()
    fusion_model = torch.nn.DataParallel(fusion_model).cuda()
    wandb.watch(model)
    wandb.watch(fusion_model)

    val_data = Action_DATASETS(config.data.val_list, config.data.label_list, num_segments=config.data.num_segments,
                        image_tmpl=config.data.image_tmpl,
                        transform=transform_val, random_shift=config.random_shift)
    val_loader = DataLoader(val_data, batch_size=config.data.batch_size, num_workers=config.data.workers, shuffle=False,
                            pin_memory=True, drop_last=True)

    if device == "cpu":
        model_text.float()
        model_image.float()
    else:
        clip.model.convert_weights(
            model_text)  # Actually this line is unnecessary since clip by default already on float16
        clip.model.convert_weights(model_image)

    start_epoch = config.solver.start_epoch

    if config.pretrain:
        if os.path.isfile(config.pretrain):
            print(("=> loading checkpoint '{}'".format(config.pretrain)))
            checkpoint = torch.load(config.pretrain)
            model.load_state_dict(checkpoint['model_state_dict'])
            fusion_model.load_state_dict(checkpoint['fusion_model_state_dict'])
            del checkpoint
        else:
            print(("=> no checkpoint found at '{}'".format(config.pretrain)))

    classes, num_text_aug, text_dict = text_prompt(val_data)

    best_prec1 = 0.0
    prec1 = validate(start_epoch, val_loader, classes, device, model, fusion_model, config, num_text_aug, args.save_output)

if __name__ == '__main__':
    main()
