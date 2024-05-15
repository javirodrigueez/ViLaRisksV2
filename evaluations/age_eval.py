"""
Script to evaluate age detection capabilities of SEVILA. 

Usage: 
    age_eval_temp.py <videosDir> <videosFile> <outFile>

Options:
    -h --help       Show this screen.

Arguments:
    <videosDir>     Path to directory with video files.
    <videosFile>    Path to the file containing videos (annotated). ETRI path is hardcoded.
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
import warnings
import sys

## Aux functions
letters_dict = { 0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}
def get_options():
  options = ['Less than 15 years', 'Between 15 and 65 years', 'More than 65 years']
  letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
  template = 'Option {}:{}.'
  list = [template.format(letters[i], options[i]) for i in range(len(options))]
  return options, ' '.join(list)

args = docopt(__doc__)

## Custom vars
question2 = 'How old is the person? Focus on the face of the person.'
video_frame_num = 64
keyframe_num = 5

## Init vars
img_size = 448
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
LOC_propmpt = 'Does the information within the frames provide the necessary details to accurately answer the given question?'
QA_prompt = 'Considering the information presented in the frames, select the correct answer from the options.'
# options and inputs
option_dict, options = get_options()
text_input_qa2 = 'Question: ' + question2 + ' ' + options + ' ' + QA_prompt
text_input_loc2 = 'Question: ' + question2 + ' ' + options + ' ' + LOC_propmpt
# processors config
mean = (0.48145466, 0.4578275, 0.40821073)
std = (0.26862954, 0.26130258, 0.27577711)
normalize = transforms.Normalize(mean, std)
image_size = img_size
transform = transforms.Compose([ToUint8(), ToTHWC(), transforms_video.ToTensorVideo(), normalize])

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

## Read file and extract videos
with open(args['<videosFile>'], 'r') as f:
  videos = f.readlines()
videos = [x.strip() for x in videos]
videos = [x.split(',')[0] for x in videos]
videos.pop(0)

## Process videos
f = open(args['<outFile>'], 'w')
f.write('video,age_group\n')
for video in videos:
    # Load video
    vpath = os.path.join(args['<videosDir>'], "{}.mp4".format(video))
    if video[:2] == 'P00':
        vpath = os.path.join('/etri', "{}.mp4".format(video))
    raw_clip, indice, fps, vlen = load_video_demo(
        video_path=vpath,
        n_frms=int(video_frame_num),
        height=image_size,
        width=image_size,
        sampling="uniform",
        clip_proposal=None
    )

    # Prepare video and inputs
    clip = transform(raw_clip.permute(1,0,2,3))
    clip = clip.float().to(device)
    clip = clip.unsqueeze(0)

    # Inference
    out = sevila.generate_demo(clip, text_input_qa2, text_input_loc2, int(keyframe_num))
    scene = out['output_text'][0]
    f.write('{},{}\n'.format(video, letters_dict[scene]))
    print(video + " processed.")
    print(option_dict[scene])
    