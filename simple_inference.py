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

## Custom vars
video = '/etri/P001-P010/P001/A055_P001_G003_C002.mp4'
question = 'What action is doing the person?'
option1 = 'Less than 15 years'
option2 = 'Between 15 and 65 years'
option3 = 'More than 65 years'
video_frame_num = 32
keyframe_num = 4

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
QA_prompt = 'Considering the information presented in the frame, select the correct answer from the options.'
QA_prompt2 = 'Considering the information presented in the frames, answer correctly the question.'
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
    frame_num=4,
    answer_num=answer_num,
    task=task,
        ).to(device)

sevila.load_checkpoint(url_or_filename='https://huggingface.co/Shoubin/SeViLA/resolve/main/sevila_pretrained.pth')
print('Model Loaded')

## Load video
raw_clip, indice, fps, vlen = load_video_demo(
    video_path=video,
    n_frms=int(video_frame_num),
    height=image_size,
    width=image_size,
    sampling="uniform",
    clip_proposal=None
)

## Prepare video and inputs
clip = transform(raw_clip.permute(1,0,2,3))
clip = clip.float().to(device)
clip = clip.unsqueeze(0)
# checks
if option1[-1] != '.':
    option1 += '.'
if option2[-1] != '.':
    option2 += '.' 
if option3[-1] != '.':
    option3 += '.'
option_dict = {0:option1, 1:option2, 2:option3}
options = 'Option 1:{} Option 2:{} Option 3:{}'.format(option1, option2, option3)
text_input_qa = 'Question: ' + question + ' ' + options + ' ' + QA_prompt
text_input_qa2 = 'Question: ' + question + ' ' + QA_prompt2
text_input_loc = 'Question: ' + question + ' ' + options + ' ' + LOC_propmpt
text_input_loc2 = 'Question: ' + question + ' ' + LOC_propmpt

## Inference
out = sevila.generate_demo(clip, text_input_qa2, text_input_loc2, int(keyframe_num))

## Extra
answer_id = out['output_text'][0]
answer = option_dict[answer_id]
select_index = out['frame_idx'][0]

#print('Answer: ' + answer)
#print('Frames: ' + str(select_index))
print('Answer: ' + out['output_sequence'])