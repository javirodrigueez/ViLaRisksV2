"""
Script to evaluate scene detection capabilities of SEVILA

Usage:
    scene_eval.py <videosDir> <videosFile> <scenesFile> <outFile>

Options:
    -h --help    Show this screen.
    <videosDir>  Path to the directory containing videos.
    <videosFile>  Path to the file containing videos.
    <scenesFile> Path to the file containing the scenes.
    <outFile>    Path to the output file.
"""

from docopt import docopt
import os
import torch
from torchvision import transforms
from lavis.processors import transforms_video
from lavis.datasets.data_utils import load_video_demo
from lavis.processors.blip_processors import ToUint8, ToTHWC
from lavis.models.sevila_models.sevila import SeViLA
from typing import Optional
from charades_dataset import CharadesDataset
from torch.utils.data import DataLoader
import warnings
import sys

## Aux functions
def get_options(file_path):
  with open(file_path, 'r') as f:
    options = f.readlines()
  options = [x.strip() for x in options]
  letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
  template = 'Option {}:{}.'
  list = [template.format(letters[i], options[i]) for i in range(len(options))]
  return options, ' '.join(list)

args = docopt(__doc__)

## Custom vars
question2 = 'In which room of the house is the person?'
video_frame_num = 32
keyframe_num = 5

## Init vars
img_size = 224
num_query_token = 32
t5_model = 'google/flan-t5-xl'
drop_path_rate = 0
use_grad_checkpoint = False
vit_precision = "fp16"
freeze_vit = True
prompt = ''
max_txt_len = 77
answer_num = 5
apply_lemmatizer = False
task = 'freeze_loc_freeze_qa_vid'
# prompts
LOC_propmpt = 'Does the information within the frame provide the necessary details to accurately answer the given question?'
QA_prompt = 'Considering the information presented in the frames, select the correct answer from the options.'
# options and inputs
option_dict, options = get_options(args['<scenesFile>'])
text_input_qa2 = 'Question: ' + question2 + ' ' + options + ' ' + QA_prompt
text_input_loc2 = 'Question: ' + question2 + ' ' + options + ' ' + LOC_propmpt
# processors config
mean = (0.48145466, 0.4578275, 0.40821073)
std = (0.26862954, 0.26130258, 0.27577711)
normalize = transforms.Normalize(mean, std)
image_size = img_size
transform = transforms.Compose([ToUint8(), ToTHWC(), transforms_video.ToTensorVideo(), normalize])

## Prepare dataset
dataset = CharadesDataset(args['<videosFile>'], image_size=img_size, video_frame_num=video_frame_num, filter=True, filter_file=args['<scenesFile>'])
dataloader = DataLoader(dataset, batch_size=1, num_workers=2, shuffle=False)

## Load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Model Loading')
sevila = SeViLA(
    img_size=img_size,
    drop_path_rate=drop_path_rate,
    use_grad_checkpoint=use_grad_checkpoint,
    vit_precision=vit_precision,
    freeze_vit=freeze_vit,
    num_query_token=num_query_token,
    t5_model=t5_model,
    prompt=prompt,
    max_txt_len=max_txt_len,
    apply_lemmatizer=apply_lemmatizer,
    frame_num=keyframe_num,
    answer_num=answer_num,
    task=task,
        ).to(device)
sevila.load_checkpoint(url_or_filename='https://huggingface.co/Shoubin/SeViLA/resolve/main/sevila_pretrained.pth')
print('Model Loaded')

## Process videos
f = open(args['<outFile>'], 'w')
f.write('video,scene\n')
prec_counter = 0
for video_name, clip, y_scene in dataloader:
    # Prepare video and inputs
    clip = clip.to(device)
    # Inference
    out = sevila.generate_demo(clip, text_input_qa2, text_input_loc2, int(keyframe_num))
    scene_pred = out['output_text'][0]
    f.write('{}, {}\n'.format(video_name[0], option_dict[scene_pred]))
    print(video_name[0] + ' processed.')
    print(option_dict[scene_pred])
    if option_dict[scene_pred] == y_scene[0]:
        prec_counter += 1

print('\nPrecision: {}'.format(prec_counter/len(dataloader)))
    