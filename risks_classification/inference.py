"""
Script to do inference of risks classif model.

Usage:
    test.py <model_path>

Options:
    -h --help    Show this screen.

Arguments:
    <model_path>    Path to the model directory.
"""

from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertConfig
import torch
from docopt import docopt
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Init
args = docopt(__doc__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = args['<model_path>']
idx_to_label = {
    0: 'fall',
    1: 'burn',
    2: 'energy_wasting',
    3: 'water_wasting',
    4: 'none_risk'
}

# Load model
tokenizer = AutoTokenizer.from_pretrained(model_path)
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
model.load_state_dict(torch.load(f'{model_path}/best_model.pth'))
model.to(device)

# Inference
model.eval()
while True:
    description = input('Enter a description: ')
    if description == 'exit':
        break
    inputs = tokenizer(description, return_tensors='pt', padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    _, predicted = torch.max(outputs.logits, 1)
    print(f'Predicted: {idx_to_label[predicted.item()]}')