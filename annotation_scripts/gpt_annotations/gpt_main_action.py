"""
Script to filter captions using GPT-3.5. Script returns file containing IDs from videos.

Usage:
  gpt_main_action.py <annotationsFile> <outFile>

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
                    "text": "\"" + annotation + "\". Which is the main action of the caption? Strictly select only one of the next options. A:awaken, B:close, C:cook, D:dress, E:drink, F:eat, G:fix, H:grasp, I:hold, J:laugh, K:lie, L:make, M:open, N:photograph, O:play, P:pour, Q:put, R:run, S:sit, T:smile, U:sneeze, V:snuggle, W:stand, X:take, Y:talk, Z:throw, a:tidy, b:turn, c:undress, d:walk, e:wash, f:watch, g:work'. Strictly answer only with one of the options, don't add any auxiliar text."
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

## Get responses
total_calls = 0
f = open(outFile, "w") 
f.write('video,main_action' + '\n')
for a,v in zip(annotations, videos):
    params = create_api_call(a)
    result = client.chat.completions.create(**params)
    response = result.choices[0].message.content
    f.write(f'{v},{response}' + '\n')
    total_calls += 1
    print(v + ' has been written to file. Total calls: {}'.format(total_calls))
f.close()