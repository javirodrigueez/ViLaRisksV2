"""
Script to get contextual understanding of a scene using SEVILA.

Usage:
  risks_inferences.py <vpath> <outfile> [options]

Arguments:
    <input>             Path to the input file.
    <output>            Path to the output file.

Options:
    -h --help                   Show this screen.
    --nframes=<nframes>         Number of frames to sample from the video [default: 32].
    --keyframes=<keyframes>     Number of keyframes to use [default: 4].
"""

import os
import torch
from torchvision import transforms
from lavis.processors import transforms_video
from lavis.datasets.data_utils import load_video_demo
from lavis.processors.blip_processors import ToUint8, ToTHWC
from lavis.models.sevila_models.sevila_updated import SeViLAFeatures
from typing import Optional
import warnings
import sys
from docopt import docopt
import csv


def get_options(options):
  letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
  template = 'Option {}:{}.'
  list = [template.format(letters[i], options[i]) for i in range(len(options))]
  return ' '.join(list)

def main(args):
    ## Vars
    print('Initialising configuration')
    video = args['<vpath>']
    video_frame_num = int(args['--nframes'])
    keyframe_num = int(args['--keyframes'])
    questions = [
        ('How old is the person?', ['Adult.', 'Elder.']),
        ('In which room of the house is the person?', ['Kitchen.', 'Living room.', 'Bedroom.', 'Bathroom.', 'Dining room.'])
    ]

    ## Init conf
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
    use_vit = True
    # prompts
    LOC_propmpt = 'Does the information within the frame provide the necessary details to accurately answer the given question?'
    QA_prompt = 'Considering the information presented in frames, select the correct answer from the options.'
    # processors config
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    normalize = transforms.Normalize(mean, std)
    image_size = img_size
    transform = transforms.Compose([ToUint8(), ToTHWC(), transforms_video.ToTensorVideo(), normalize])

    ## Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Model Loading')
    sevila = SeViLAFeatures(
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
        use_vit=use_vit,
        total_frames=video_frame_num,
        device=device
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
    clip = transform(raw_clip.permute(1,0,2,3))
    clip = clip.float().to(device)
    clip = clip.unsqueeze(0)
    
    ## Inference
    file = open(args['<outfile>'], 'w')
    for q,opt in questions:
        question = q
        if not opt:
            options_str = ''
            QA_prompt = 'Considering the information presented in frames, answer correctly the question.'
        else:
            options = [o if o[-1]=='.' else f'{o}.' for o in opt]
            options_str = get_options(options)
        text_input_qa = 'Question: ' + question + ' ' + options_str + ' ' + QA_prompt
        text_input_loc = 'Question: ' + question + ' ' + options_str + ' ' + LOC_propmpt
        out = sevila.generate_demo(clip, text_input_qa, text_input_loc, int(keyframe_num))
        if opt:
            answer_id = out['output_text'][0]
            answer = options[answer_id]
        else:
            answer = out['output_sequence']
        select_index = out['frame_idx'][0]
        # show answer
        print('Question: ' + question)
        print('Answer: ' + answer)
        print('Frames: ' + str(select_index) + '\n')
        # write to csv
        writer = csv.writer(file)
        writer.writerow([question, answer, ";".join([str(i) for i in select_index])])
    file.close()

    
    

if __name__ == '__main__':
    args = docopt(__doc__)
    main(args)