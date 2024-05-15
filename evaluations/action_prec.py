"""
Script to get precision of action recognition based on cosine similarity

Usage: 
    action_prec.py <predictionFile> <gtFile> <actionsFile>

Options:
    -h --help            Show this screen.
    <predictionFile>     Path to the file containing the predictions.
    <gtFile>             Path to the file containing the ground truth.
    <actionsFile>        Path to the file containing the actions.
"""

from docopt import docopt
from evaluate import load
from sentence_transformers import SentenceTransformer, util
import torch
import csv
import numpy as np

# Init
args = docopt(__doc__)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load predictions, ground truth and actions
with open(args['<predictionFile>'], 'r') as predFile:
    predLines = predFile.readlines()
    predLines = [x.strip() for x in predLines]
    predActions = [x.split(',')[1] for x in predLines]
    predActions.pop(0)

with open(args['<gtFile>'], 'r') as gtFile:
    csv_reader = csv.reader(gtFile)
    next(csv_reader)    # Skip header
    gtActions = []
    removals = []
    counter = 0
    for row in csv_reader:
        if row[9] == '':
            removals.append(counter)
            continue
        actions = row[9].split(';')
        actions = [i.split()[0] for i in actions]
        gtActions.append(actions)
        counter += 1

with open(args['<actionsFile>'], 'r') as actionsFile:
    actions = actionsFile.readlines()
    actions = [x.strip() for x in actions]    
    actions_dict = {i.split()[0]: " ".join(i.split()[1:]) for i in actions}

actions_emb_list = model.encode(list(actions_dict.values()), convert_to_tensor=True)
actions_emb_dict = {i: actions_emb_list[j] for j,i in enumerate(actions_dict.keys())}

# Delete rows with empty ground truth
for i in reversed(removals):
    predActions.pop(i)
    
# Evaluate prediction
correct = 0
for pred,gt in zip(predActions,gtActions):
    gtList = []
    for i in gt:
        gtList.append(actions_dict[i])
    pred_emb = model.encode(pred.strip())
    max_cos_sim = 0
    max_key = ""
    # Classify prediction into one of the actions doing cosine similarity
    predList = torch.tensor(np.array([pred_emb] * len(actions_emb_list))).to('cuda')
    # predList = torch.full((1, len(actions_emb_list)), pred_emb)
    cosine_scores = util.cos_sim(predList, actions_emb_list)
    max_idx = cosine_scores.argmax()
    pred_class = list(actions_dict.values())[max_idx]
    # Search for match
    for i in gtList:
        if i == pred_class:
            correct += 1
            break

# Calculate precision
precision = correct / len(predActions)
print(f"Precision: {precision}")