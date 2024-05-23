"""
Script to generate description from items extracted from a video and classify description into a risk

Usage:
  gpt_risks_descriptions.py --objects_file=<objects_file> --answer_file=<answer_file> [options] --label_map=<label_map> --verbs_map=<verbs_map>

Options:
    -h --help               Show this screen.
    --llm=<llm>             Choose LLM to use: gpt3-5|t5|mistral|gemma [default: gpt3-5]
    --ra_model=<ra_model>   Path to risk assessment model [default: risks_classification/risks_classifier_model]

Arguments:
    <objects_file>      Path to the file containing the objects.
    <answer_file>       Path to the file containing the answers.
    <label_map>         Path to the label map.
    <verbs_map>         Path to the verbs map.
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
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import T5TokenizerFast, T5Config, T5ForConditionalGeneration
from transformers import AutoModelForCausalLM
import math
import gensim.downloader as api

## Functions
def calcular_centro(bbox):
    # bbox is a tuple (x1, y1, x2, y2)
    x1, y1, x2, y2 = bbox
    centro_x = (x1 + x2) / 2
    centro_y = (y1 + y2) / 2
    return centro_x, centro_y

def distancia_euclidiana(bbox1, bbox2):
    # Obtain bbox centers
    c1_x, c1_y = calcular_centro(bbox1)
    c2_x, c2_y = calcular_centro(bbox2)
    
    # Compute euclidean distance from centers
    distancia = math.sqrt((c1_x - c2_x)**2 + (c1_y - c2_y)**2)
    return distancia

def get_main_object(bboxes):
    global label_map
    # get object with label person
    person_boxes = []
    for bbox in bboxes:
        obj_str = label_map[str(int(bbox[5].item()))]
        if obj_str == 'person':
            person_boxes.append(bbox)
    # get object with minimum distance to person
    distances = []
    main_objects = []
    for person_box in person_boxes:
        min_dist = float('inf')
        main_obj = None
        for bbox in bboxes:
            obj_str = label_map[str(int(bbox[5].item()))]
            if obj_str != 'person' and person_box[6] == bbox[6]:
                dist = distancia_euclidiana(person_box[:4], bbox[:4])
                if dist < min_dist:
                    min_dist = dist
                    main_obj = bbox
        distances.append(min_dist)
        main_objects.append(main_obj)
    min_dist = min(distances)
    main_obj = main_objects[distances.index(min_dist)]
    return main_obj

def get_main_action(obj, actions):
    # get action with maximum similarity to main object
    print('loading word2vec...')
    wv = api.load('word2vec-google-news-300')
    max_sim = 0
    main_action = None
    for action in actions:
        sim = wv.similarity(obj, action)
        if sim > max_sim:
            max_sim = sim
            main_action = action
    return main_action

def create_t5_model(t5_model):
    t5_tokenizer = T5TokenizerFast.from_pretrained(t5_model)
    t5_config = T5Config.from_pretrained(t5_model)
    t5_config.dense_act_fn = "gelu"
    t5_model = T5ForConditionalGeneration.from_pretrained(t5_model, config=t5_config)
    return t5_tokenizer, t5_model

def create_api_call(prompt):
    prompt_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
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
verbs_map = args['--verbs_map']
llm = args['--llm']
ra_model = args['--ra_model']
client = OpenAI(api_key=os.getenv('OPEN_API_KEY'))
risks_labels_map = {0: 'fall', 1: 'burn', 2: 'energy_wasting', 3: 'water_wasting', 4: 'none_risk'}

## Read files
# gdino objects
with open(label_map, 'r') as file:
    label_map = json.load(file)
with open(verbs_map, 'r') as file:
    verbs_map = json.load(file)
all_objects = torch.load(objects_file)
thresh_obj = [[obj if obj[4] > 0.25 else None for obj in frame] for frame in all_objects['res_info']]
thresh_obj = [obj for frame in thresh_obj for obj in frame]
thresh_obj = list(filter(lambda x: x is not None, thresh_obj))
main_obj = get_main_object(thresh_obj)
print('Main object data: ', main_obj)
main_obj_str = label_map[str(int(main_obj[5].item()))]
if '_' in main_obj_str:
    main_obj_str = main_obj_str.split('_')[0]
print('Main object label:', main_obj_str, '\n')
# sevila and actionclip answers
with open(answer_file, 'r') as file:
    reader = csv.reader(file)
    answers = list(reader)
    age = answers[0][1]
    room = answers[1][1]
    actions = answers[2][1]
actions = actions.split(';')
actions_str = [verbs_map[action] for action in actions]
print('Top5 actions:', actions_str, '\n')
# compute cosine similarity between main object and top5 actionclip actions
main_action = get_main_action(main_obj_str, actions_str)
print('Main action:', main_action, '\n')

## Get responses
prompt = "Create a brief technical description in a phrase of a home scene including the object: {}, the action: {}, the  age of the person: {}, and the scene: {}.".format(main_obj_str, main_action, age.strip('.'), room.strip('.'))
if llm == 'gpt3-5':
    params = create_api_call(prompt)
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
# define model
tokenizer = AutoTokenizer.from_pretrained(ra_model)
ckpt = 'distilbert/distilbert-base-uncased-finetuned-sst-2-english'
model = AutoModelForSequenceClassification.from_pretrained(ckpt)
in_features = model.classifier.in_features
model.classifier = torch.nn.Sequential(
    torch.nn.LayerNorm(in_features),
    torch.nn.Linear(in_features, 368),
    torch.nn.ReLU(),
    torch.nn.Linear(368, 5)
)
# load weights
model.load_state_dict(torch.load(f'{ra_model}/best_model.pth'))
model.to(device)
model.eval()

# inference
inputs = tokenizer(response, return_tensors='pt', padding=True, truncation=True)
inputs = {k: v.to(device) for k, v in inputs.items()}
outputs = model(**inputs)
_, predicted = torch.max(outputs.logits, 1)
risk = risks_labels_map[predicted.item()]
print('Generated description:', response, '\n')
print('Risk:', risk, '\n')
