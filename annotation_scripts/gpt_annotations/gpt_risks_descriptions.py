"""
Script to generate synthetic scenes descriptions from GPT3.5

Usage:
  gpt_risks_descriptions.py

Options:
    -h --help     Show this screen.
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

## Constants
number = 10
room = '\"kitchen\"'
risk = '\"none risk\"'
example = '\"In the kitchen, a person is washing dishes in the sink, their hands moving efficiently as they scrub away grime. Nearby, a pot of tea steeps on the countertop, releasing its fragrant aroma into the air. On the shelf, a stack of plates sits next to a container of dish soap, awaiting their turn to be cleaned.\"'

## Functions
def create_api_call():
    prompt_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Create strictly {} descriptions separated by an endline with the next instructions. Create a technical description of a home scene including the word {}, the action being performed by a person and 4-8 objects that are in the room. Align the description with the risk {} without mentioning the risk. Maintain the same structure in each of your descriptions. Next I give you an example: {}"
                        .format(number, room, risk, example)
                }
            ]
        }
    ]
    params = {
        "model": "gpt-3.5-turbo",
        "messages": prompt_messages,
        "max_tokens": 2000,
    }
    return params

## Init environment
args = docopt(__doc__)
load_dotenv()
client = OpenAI(api_key=os.getenv('OPEN_API_KEY'))

## Get responses
# print('room\trisk\tdescription')
for i in range(5):
    params = create_api_call()
    result = client.chat.completions.create(**params)
    response = result.choices[0].message.content
    descriptions = response.split('\n')
    descriptions = list(filter(None, descriptions))             # remove empty strings
    for d in descriptions:
        print(f'{room.strip('"')}\t{risk.strip('"')}\t{d}')     # tsv format