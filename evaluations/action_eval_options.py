"""
Script to evaluate action detection capabilities of SEVILA. Output action is an option chosen by the language model.

Usage:
    action_eval.py <verbsFile> <videosFile> <outFile>

Options:
    -h --help    Show this screen.

Arguments:
    <verbsFile>     Path to the file containing verbs.
    <videosFile>    Path to the file containing videos.
    <outFile>       Path to the output file.
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
letters_dict = { 0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'a', 27: 'b', 28: 'c', 29: 'd', 30: 'e', 31: 'f', 32: 'g', 33: 'h', 34: 'i', 35: 'j', 36: 'k', 37: 'l', 38: 'm', 39: 'n', 40: 'o', 41: 'p', 42: 'q', 43: 'r', 44: 's', 45: 't', 46: 'u', 47: 'v', 48: 'w', 49: 'x', 50: 'y', 51: 'z'}
def get_options(file_path):
  with open(file_path, 'r') as f:
    options = f.readlines()
  options = [x.strip() for x in options]
  letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
  template = 'Option {}:{}.'
  list = [template.format(letters[i], options[i]) for i in range(len(options))]
  return options, ' '.join(list)

args = docopt(__doc__)

## Custom vars
question = 'What is doing the person?'  
# question = 'Which 5 objects appear in the video?'
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
LOC_prompt = 'Does the information within the frame provide the necessary details to accurately answer the given question?'
QA_prompt = 'Considering the information presented in the frames, answer correctly the question.'
# options and inputs
option_dict, options = get_options(args['<verbsFile>'])
text_input_qa = 'Question: ' + question + ' ' + options + ' ' + QA_prompt
print(text_input_qa)
text_input_loc = 'Question: ' + question + ' ' + options + ' ' + LOC_prompt
# processors config
mean = (0.48145466, 0.4578275, 0.40821073)
std = (0.26862954, 0.26130258, 0.27577711)
normalize = transforms.Normalize(mean, std)
image_size = img_size
transform = transforms.Compose([ToUint8(), ToTHWC(), transforms_video.ToTensorVideo(), normalize])

## Prepare dataset
dataset = CharadesDataset(args['<videosFile>'], image_size=img_size, video_frame_num=video_frame_num, field=1)
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
f.write('video,action\n')
prec_counter = 0
for video_name, clip, _ in dataloader:
    # Prepare video and inputs
    clip = clip.to(device)
    # Inference
    out = sevila.generate_demo(clip, text_input_qa, text_input_loc, int(keyframe_num))
    action_pred = out['output_text'][0]
    f.write('{}, {}\n'.format(video_name[0], letters_dict[action_pred]))
    print(video_name[0] + ' processed.')
    print(option_dict[action_pred])
    