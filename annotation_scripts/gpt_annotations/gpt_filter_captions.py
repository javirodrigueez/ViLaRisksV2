"""
Script to filter captions using GPT-3.5. Script returns file containing IDs from videos.

Usage:
  gpt_filter_captions.py <annotationsFile> <outFile>

Options:
    -h --help           Show this screen.
    <annotationsFile>   Path to the file containing annotations.
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

def create_api_call(annotation):
    prompt_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "\"" + annotation + "\". Does the caption suggests that more than one person intervene in the video? A: Yes, B: No. Select the correct answer from the options."
                }
            ]
        }
    ]
    params = {
        "model": "gpt-3.5-turbo",
        "messages": prompt_messages,
        "max_tokens": 1,
    }
    return params

## Init
args = docopt(__doc__)
load_dotenv()
client = OpenAI(api_key=os.getenv('OPEN_API_KEY'))

## Read file and extract captions
annotationsFile = args['<annotationsFile>']
outFile = args['<outFile>']
with open(annotationsFile, 'r') as file:
    data = file.readlines()
data = [x.strip() for x in data]
data = [x.split(',') for x in data]
annotations = [x[6] for x in data]
videos = [x[0] for x in data]
annotations.pop(0)
videos.pop(0)

## Get responses
counter = 0
total_calls = 0
f = open(outFile, "w") 
for a,v in zip(annotations, videos):
    params = create_api_call(a)
    result = client.chat.completions.create(**params)
    response = result.choices[0].message.content
    if response == 'B':
        f.write(v + '\n')
        print(v + ' has been written to file.')
        counter += 1
    total_calls += 1
    print('Total calls: {}, Accepted: {}'.format(total_calls, counter))
f.close()