"""
Script to test risks classif model.

Usage:
    test.py <model_path> <testset> [options]

Options:
    -h --help    Show this screen.

Arguments:
    <model_path>    Path to the model directory.
    <testset>    Path to the test data. Expected in TSV format with columns: label, description.
"""

from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertConfig
import torch
from docopt import docopt
from tqdm import tqdm
from risks_dataset import RisksDataset
from torch.utils.data import DataLoader

# Init
args = docopt(__doc__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = args['<model_path>']

# Load model
config = DistilBertConfig.from_pretrained(model_path, num_labels=5)
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path, config=config)
model.to(device)

# Load data
data_path = args['<testset>']
test_dataset = RisksDataset(data_path, tokenizer)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model.eval()
with torch.no_grad():
    total = 0
    correct = 0
    for batch in tqdm(test_dataloader):
        inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}  # Move batch to device
        labels = batch['labels'].to(device)
        outputs = model(**inputs)
        _, predicted = torch.max(outputs.logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(f'Accuracy: {correct / total}')