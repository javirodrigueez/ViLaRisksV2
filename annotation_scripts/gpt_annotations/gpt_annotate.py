"""
Usage:
  gpt_annotate.py <keyframesFile> <videosDir> <outFile>

Options:
    -h --help           Show this screen.
    <keyframesFile>     Path to the file videos and keyframes.
    <videosDir>         Path to the directory containing videos.
    <outFile>           Path to the output file.
"""

import cv2
import base64
import time
from openai import OpenAI
import os
import requests
import numpy as np
from dotenv import load_dotenv
from docopt import docopt
from lavis.datasets.data_utils import load_video_demo

def create_api_call(frames):
    prompt_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "How old is the person? A: Less than 5 years, B: Between 5 and 15 years, C: Between 15 and 45 years, D: Between 45 and 70 years, E: More than 70 years. Considering the information in frames, select the correct answer from the options. Answer only with the option"
                },
                {
                    "type": "img_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{frames[0]}",
                        "detail": "low"
                    }
                },
                {
                    "type": "img_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{frames[1]}",
                        "detail": "low"
                    }
                },
                {
                    "type": "img_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{frames[2]}",
                        "detail": "low"
                    }
                },
                {
                    "type": "img_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{frames[3]}",
                        "detail": "low"
                    }
                },
                {
                    "type": "img_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{frames[4]}",
                        "detail": "low"
                    }
                }
            ]
        },
    ]
    params = {
        "model": "gpt-4-vision-preview",
        "messages": prompt_messages,
        "max_tokens": 1,
    }
    return params

## Init
load_dotenv()
client = OpenAI(api_key=os.getenv('OPEN_API_KEY'))
args = docopt(__doc__)
video_frame_num = 64
image_size = 448
out = open(args['<outFile>'], 'w')
out.write('video,age_group\n')

## Load keyframes
with open(args['<keyframesFile>'], 'r') as file:
    keyframes = file.readlines()
    keyframes.pop(0)
    keyframes = [x.strip() for x in keyframes]
    keyframes = [x.split(',') for x in keyframes]

## Open videos and extract frames
for keyframe in keyframes:
    # obtain video
    video = keyframe[0]
    keyframe_list = keyframe[1]
    keyframe_list = keyframe_list.split(';') 
    keyframe_list = [int(x) for x in keyframe_list]
    vpath = os.path.join(args['<videosDir>'], "{}.mp4".format(video))
    raw_clip, indice, fps, vlen = load_video_demo(
        video_path=vpath,
        n_frms=int(video_frame_num),
        height=image_size,
        width=image_size,
        sampling="uniform",
        clip_proposal=None
    )
    # obtain and process frames
    frames = []
    for i in keyframe_list:
        frame = raw_clip[:, i, :, :].int().permute(1,2,0).numpy()
        _, buffer = cv2.imencode(".jpg", frame)
        frames.append(base64.b64encode(buffer).decode("utf-8"))
    # api call
    params = create_api_call(frames)
    result = client.chat.completions.create(**params)
    response = result.choices[0].message.content
    out.write(video + ',' + response + '\n')
    print(video + " processed.")

out.close()