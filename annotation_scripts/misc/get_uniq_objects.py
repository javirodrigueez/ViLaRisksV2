"""
Script to get unique objects from trainset

Usage:
    get_uniq_objects.py <csvFile>

Options:
    -h --help    Show this screen.
    <csvFile>  Path to the file containing objects.
"""


import csv
from docopt import docopt

args = docopt(__doc__)

elementos_unicos = set()

with open(args['<csvFile>'], mode='r', encoding='utf-8') as csvfile:
    csv_reader = csv.reader(csvfile)
    
    for row in csv_reader:
        septimo_campo = row[7]
        
        elementos = septimo_campo.split(';')
        
        elementos_unicos.update(elementos)

for elemento in elementos_unicos:
    print(elemento)
