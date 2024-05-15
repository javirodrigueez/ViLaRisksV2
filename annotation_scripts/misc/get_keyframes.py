"""
Script to extract keyframes from videos using SeViLA model.

Usage:
    get_keyframes.py <videosDir> <videosFile> <outFile>

Options:
    -h --help    Show this screen.
    <videosDir>  Path to the directory containing videos.
    <videosFile>  Path to the file containing videos.
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
import warnings
import sys

args = docopt(__doc__)

## Custom vars
question2 = 'Which is the age and the gender of the person?'
option1 = 'Yes'
option2 = 'No'
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
LOC_propmpt = 'Does the information within the frame provide the necessary details to accurately answer the given question?'
QA_prompt = 'Considering the information presented in the frame, select the correct answer from the options.'
# options and inputs
if option1[-1] != '.':
    option1 += '.'
if option2[-1] != '.':
    option2 += '.' 
option_dict = {0:option1, 1:option2}
options = 'Option A:{} Option B:{}'.format(option1, option2)
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
with open(args['<videosFile>'], 'r') as file:
    videos = file.readlines()
videos = [x.strip() for x in videos]


## Process videos
f = open(args['<outFile>'], 'w')
f.write('video,keyframes\n')
counter = 0
for video in videos:
    # If 200 videos has been collected, stop
    if counter == 200:
        break
    # Load video
    vpath = os.path.join(args['<videosDir>'], "{}.mp4".format(video))
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

    # Inference 1 (Keyframes for age/gender estimation)
    out = sevila.generate_demo(clip, text_input_qa2, text_input_loc2, int(keyframe_num))
    select_index = out['frame_idx'][0]
    f.write(video + ',' + ';'.join([str(i) for i in select_index]) + '\n')
    counter += 1
    print(video + " processed.")
    