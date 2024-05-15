"""
Script to generate description from items extracted from a video and classify description into a risk

Usage:
  gpt_risks_descriptions.py --objects_file=<objects_file> --answer_file=<answer_file> [options] --label_map=<label_map>

Options:
    -h --help               Show this screen.
    --llm=<llm>             Choose LLM to use: gpt3-5|t5|mistral|gemma [default: gpt3-5]
    --ra_model=<ra_model>   Path to risk assessment model [default: risks_classification/risks_classifier_model]

Arguments:
    <objects_file>      Path to the file containing the objects.
    <answer_file>       Path to the file containing the answers.
    <label_map>         Path to the label map.
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
import torch
import csv, json
from transformers import DistilBertConfig, DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import T5TokenizerFast, T5Config, T5ForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForCausalLM

## Functions
def create_t5_model(t5_model):
    t5_tokenizer = T5TokenizerFast.from_pretrained(t5_model)
    t5_config = T5Config.from_pretrained(t5_model)
    t5_config.dense_act_fn = "gelu"
    t5_model = T5ForConditionalGeneration.from_pretrained(t5_model, config=t5_config)
    return t5_tokenizer, t5_model
def create_api_call(objects, room, description, example):
    prompt_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Objects: {}. Room: {}. Description: {}. Create a technical description of a home scene from the provided description, the room of the house and the objects mentioned. Don't abuse adjectives and don't make up things I haven't mentioned. Follow the structure of the next example \"{}\""
                        .format(objects, room, description, example)
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
objects_file = args['--objects_file']
answer_file = args['--answer_file']
label_map = args['--label_map']
llm = args['--llm']
ra_model = args['--ra_model']
client = OpenAI(api_key=os.getenv('OPEN_API_KEY'))
risks_labels_map = {0: 'fall', 1: 'burn', 2: 'energy_wasting', 3: 'water_wasting', 4: 'none_risk'}

## Read files
# gdino objects
with open(label_map, 'r') as file:
    label_map = json.load(file)
all_objects = torch.load(objects_file)
thresh_obj = [i if i[4] > 0.25 else None for i in all_objects['res_info'][0]]
thresh_obj = list(filter(lambda x: x is not None, thresh_obj))
thresh_obj = set([label_map[str(int(i[5]))] for i in thresh_obj])
thresh_obj_str = ', '.join(thresh_obj)
print('Objects:', thresh_obj_str, '\n')
# sevila answers
with open(answer_file, 'r') as file:
    reader = csv.reader(file)
    answers = list(reader)
    age = answers[0][1]
    room = answers[1][1]
    description = answers[2][1]

## Get responses
example = "In the kitchen, a person is searing a piece of meat on a piping hot skillet. Nearby, a pot of boiling water is ready for pasta. On the table, a container of salt sits next to a pepper grinder."
prompt = "Objects: {}. Room: {}. Description: {}. Create a description of a home scene from the provided description, the room of the house and the objects mentioned. Don't make up things. Follow the structure of the next example \"{}\".".format(thresh_obj_str, room.strip('.'), description, example)
if llm == 'gpt3-5':
    params = create_api_call(thresh_obj_str, room.strip('.'), description, example)
    result = client.chat.completions.create(**params)
    response = result.choices[0].message.content
elif llm == 't5':
    t5_tokenizer, t5_model = create_t5_model("google/flan-t5-xl")
    t5_model.to(device)
    t5_model.eval()
    input_ids = t5_tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    response = t5_model.generate(input_ids, max_new_tokens=2000)
    response = t5_tokenizer.decode(response.squeeze(), skip_special_tokens=True)
elif llm == 'gemma':
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-1.1-2b-it")
    model = AutoModelForCausalLM.from_pretrained("google/gemma-1.1-2b-it", torch_dtype=torch.bfloat16)
    model.to(device)
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)
    response = model.generate(**input_ids, max_new_tokens=2000)
    response = tokenizer.decode(response.squeeze())
elif llm == 'mistral':
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", torch_dtype=torch.bfloat16)
    model.to(device)
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)
    response = model.generate(**input_ids, max_new_tokens=2000)
    response = tokenizer.decode(response.squeeze())
    response = response.replace('<s>', '').replace('</s>', '').split('\n')[-1]
else:
    raise ValueError('LLM not supported')

## Risks assessment
config = DistilBertConfig.from_pretrained(ra_model, num_labels=5)
tokenizer = DistilBertTokenizer.from_pretrained(ra_model)
model = DistilBertForSequenceClassification.from_pretrained(ra_model, config=config)
model.to(device)
model.eval()

inputs = tokenizer(response, return_tensors='pt', padding=True, truncation=True)
inputs = {k: v.to(device) for k, v in inputs.items()}
outputs = model(**inputs)
_, predicted = torch.max(outputs.logits, 1)
risk = risks_labels_map[predicted.item()]
print('Generated description:', response, '\n')
print('Risk:', risk, '\n')
