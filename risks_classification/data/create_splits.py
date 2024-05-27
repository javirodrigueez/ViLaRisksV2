"""
Script to crate train/test splits.

Usage:
    create_splits.py <input_data> <output_dir> [options]

Options:
    -h --help    Show this screen.
    --test_size=<float>    Test/val total size [default: 0.2]
    --include_val    Include validation set.
    <input_data>    Path to the input data. Expected in TSV format with columns: room, label, description.
    <output_dir>    Path to the output dir.
"""

from sklearn.model_selection import train_test_split
from docopt import docopt
import os

args = docopt(__doc__)

# Init
data_path = args['<input_data>']
output_dir = args['<output_dir>']
test_size = float(args['--test_size'])
split_size = test_size / 2
include_val = args['--include_val']

# Load data
X = []
y = []
with open(data_path, 'r') as f:
    lines = f.readlines()
    lines = lines[1:]
    for line in lines:
        data = line.strip().split('\t')
        X.append(data[2])
        y.append(data[1])

# Create splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)
if include_val:
    X_val, X_val, y_val, y_val = train_test_split(X_test, y_test, test_size=split_size, stratify=y_test)

# Save splits
output_path_train = os.path.join(output_dir, 'risks_train.tsv')
output_path_test = os.path.join(output_dir, 'risks_test.tsv')
output_path_val = os.path.join(output_dir, 'risks_val.tsv')
with open(output_path_train, 'w') as f:
    for x, y in zip(X_train, y_train):
        f.write(f'{y}\t{x}\n')
with open(output_path_test, 'w') as f:
    for x, y in zip(X_test, y_test):
        f.write(f'{y}\t{x}\n')
if include_val:
    with open(output_path_val, 'w') as f:
        for x, y in zip(X_val, y_val):
            f.write(f'{y}\t{x}\n')
