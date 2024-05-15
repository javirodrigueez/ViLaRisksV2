"""
Script to train text classification model.

Usage:
    train.py <trainset> [options]

Options:
    -h --help    Show this screen.
    --batch_size=<int>    Batch size [default: 64]
    --model=<model>   Model to use. Possible values: distilbert|xlmroberta|bert|deberta [default: distilbert]

Arguments:
    <trainset>    Path to the training data. Expected in TSV format with columns: label, description.
"""

import torch
from torch.utils.data import DataLoader, Subset
from risks_dataset import RisksDataset
from docopt import docopt
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
import wandb

args = docopt(__doc__)
wandb.init(mode='disabled')
model_name = args['--model']

if model_name == 'distilbert':
    ckpt = 'distilbert-base-uncased'
    #ckpt = 'distilbert/distilbert-base-uncased-finetuned-sst-2-english'
elif model_name == 'xlmroberta':
    ckpt = 'xlm-roberta-base'
elif model_name == 'bert':
    ckpt = 'bert-base-uncased'
elif model_name == 'deberta':
    ckpt = 'microsoft/deberta-v3-base'
else:
    print('Invalid model name')
    exit()

# Cargar datos
data_path = args['<trainset>']
data_frame = pd.read_csv(data_path, sep='\t')
labels = data_frame['risk'].values
tokenizer = AutoTokenizer.from_pretrained(ckpt)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parámetros
batch_size = int(args['--batch_size'])
lr = 1e-5
epochs = 100
k_folds = 10
skf = StratifiedKFold(n_splits=k_folds, shuffle=True)

# Bucle de cross-validation
results = []
max_accuracy = 0.0
for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
    print(f'Training fold {fold+1}/{k_folds}')
    # Preparar datasets
    train_subsampler = RisksDataset(data_frame.iloc[train_idx], tokenizer)
    val_subsampler = RisksDataset(data_frame.iloc[val_idx], tokenizer)
    train_loader = DataLoader(train_subsampler, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subsampler, batch_size=batch_size, shuffle=False)

    # Modelo
    model = AutoModelForSequenceClassification.from_pretrained(ckpt)
    for name, param in model.named_parameters():
        if ('classifier' not in name) and ('pre_classifier' not in name) and ('pooler' not in name or ckpt=='bert-base-uncased'):
            param.requires_grad = False
    in_features = model.classifier.in_features
    model.classifier = torch.nn.Linear(in_features, 5)
    model.to(device)

    # Deberta adaptation
    if 'deberta' in ckpt:
        model.pooler.requires_grad = True 
    
    # Optimizador y pérdida
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = CrossEntropyLoss()

    # Entrenamiento
    for epoch in tqdm(range(epochs)):
        model.train()
        for batch in train_loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)
            outputs = model(**inputs)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    # Validación
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)
            outputs = model(**inputs)
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    if accuracy > max_accuracy:
        max_accuracy = accuracy
        model_path = "tests_model"
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
    print(f'Fold {fold+1} accuracy: {accuracy}')
    results.append(accuracy)

# Promediar y mostrar resultados finales
print(f'Mean accuracy across folds: {np.mean(results)}')
print(f'Max accuracy: {max_accuracy}')
wandb.finish()
