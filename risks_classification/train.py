"""
Script to train text classification model.

Usage:
    train.py <trainset> <testset> [options]

Options:
    -h --help    Show this screen.
    --batch_size=<int>    Batch size [default: 32]

Arguments:
    <trainset>    Path to the training data. Expected in TSV format with columns: label, description.
    <testset>    Path to the test data. Expected in TSV format with columns: label, description.
"""

import torch
from torch.utils.data import DataLoader
from risks_dataset import RisksDataset
from docopt import docopt
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, AdamW
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
import wandb

args = docopt(__doc__)

# Init
#wandb.init(project="risks-classification", entity="javirodriguez")
wandb.init(mode='disabled')
data_path = args['<trainset>']
data_path_val = args['<testset>']
batch_size = int(args['--batch_size'])
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare data
dataset = RisksDataset(data_path, tokenizer)
val_dataset = RisksDataset(data_path_val, tokenizer)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# Load and prepare model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
for name, param in model.named_parameters(): # freeze layers
    if 'classifier' not in name:
        param.requires_grad = False
in_features = model.classifier.in_features
model.classifier = torch.nn.Linear(in_features, 5) # adapt classifer to receive 5 classes
model.to(device)

# Hyperparams
lr = 5e-5
epochs = 100
optimizer = AdamW(model.parameters(), lr=lr)
criterion = CrossEntropyLoss()

for epoch in range(epochs):  # Number of epochs
    model.train()
    loop = tqdm(dataloader, leave=True)
    for batch in loop:
        inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}  # Move batch to device
        labels = batch['labels'].to(device)
        outputs = model(**inputs)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())

    # Validate model
    # model.eval()
    # with torch.no_grad():
    #     total = 0
    #     correct = 0
    #     for batch in tqdm(val_dataloader):
    #         inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}  # Move batch to device
    #         labels = batch['labels'].to(device)
    #         outputs = model(**inputs)
    #         _, predicted = torch.max(outputs.logits, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()
    #     print(f'Accuracy: {correct / total}')
    
    # Log results
    # wandb.log({"accuracy": correct / total, "loss": loss.item()})


# Save model
model_path = "risks_classifier_model"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

wandb.finish()