"""
Script to generate synthetic scenes descriptions from GPT3.5

Usage:
  gpt_risks_descriptions.py <input> <objects> <actions> <output>

Arguments:
    <input>       Input file with the objects, actions, age and observation.
    <output>      Output file with the generated descriptions.
    <objects>     Possible random objects.
    <actions>     Possible random actions.

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
number = 2
#room = '\"kitchen\"'
#risk = '\"none risk\"'
#example = '\"In the kitchen, a person is washing dishes in the sink, their hands moving efficiently as they scrub away grime. Nearby, a pot of tea steeps on the countertop, releasing its fragrant aroma into the air. On the shelf, a stack of plates sits next to a container of dish soap, awaiting their turn to be cleaned.\"'

## Functions
def get_items(items):
    with open(items, 'r') as f:
        items = f.readlines()
        items = [i.strip() for i in items]
    return items

def rephrase_description(description):
    seed = np.random.randint(0, 100000)
    prompt_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Rephrase the next sentence: {}".format(description)
                }
            ],
            "seed": seed,
        }
    ]
    params = {
        "model": "gpt-3.5-turbo",
        "messages": prompt_messages,
        "max_tokens": 2000,
    }
    return params


def create_api_call(objects, actions, age, observation):
    seed = np.random.randint(0, 100000)
    prompt_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Create a brief technical description in a phrase of a home scene including the object: {}, the action: {}, the  age of the person: {}, and the scene: kitchen. Also take in mind the next observation of the scene: {}"
                        .format(objects, actions, age, observation)
                }
            ],
            "seed": seed,
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

## Declare randoms
ages = ['adult', 'elder']
common_objects = get_items(args['<objects>'])
common_actions = get_items(args['<actions>'])

first = True
f_out = open(args['<output>'], 'w')
with open(args['<input>'], 'r') as f:
    lines = f.readlines()
    for line in lines:
        if first:
            first = False
            continue
        risk, specific_obj, specific_act, observation, number = line.strip().split(',')
        descriptions = []
        for i in range(int(number)):
            age = np.random.choice(ages)
            if specific_obj != 'none':
                #common_obj = np.random.choice(common_objects, size=2, replace=False)
                #objects = ';'.join(common_obj) + ';' + specific_obj
                objects = specific_obj
            else:
                #common_obj = np.random.choice(common_objects, size=3, replace=False)
                #objects = ';'.join(common_obj)
                objects = np.random.choice(common_objects)
            if specific_act != 'none':
                #common_act = np.random.choice(common_actions, size=2, replace=False)
                #actions = ';'.join(common_act) + ';' + specific_act
                actions = specific_act
            else:
                #common_act = np.random.choice(common_actions, size=3, replace=False)
                #actions = ';'.join(common_act)
                actions = np.random.choice(common_actions)
            params = create_api_call(objects, actions, age, observation)
            result = client.chat.completions.create(**params)
            response = result.choices[0].message.content
            response = response.strip('\n').strip('.')
            if response in descriptions:
                for i in range(5):
                    params = rephrase_description(response)
                    result = client.chat.completions.create(**params)
                    response = result.choices[0].message.content
                    response = response.strip('\n').strip('.')
                    if response not in descriptions:
                        descriptions.append(response)
                        break
            else:
                descriptions.append(response)
            f_out.write(f'{risk},{objects},{actions},{age},"{response}"\n')
            # descriptions = response.split('\n')
            # descriptions = list(filter(None, descriptions))             # remove empty strings
            # for d in descriptions:
            #     print(f'{risk},{objects},{actions},{age},{d}')     # csv format
        print(f'{risk},{objects},{actions} completed.')
f_out.close()
