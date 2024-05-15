"""
Script to create COCO annotations from video.

Usage:
    create_coco.py <vpath> <outdir> <catfile> [options]
    
Options:
    -h --help    Show this screen.
    --nframes=<n>    Number of frames to sample. [default: 32]

Arguments
    <vpath>      Path to video.
    <catfile>    Path to categories file.
    <outdir>     Path to output directory.
"""

from docopt import docopt
import json, glob
import numpy as np
from PIL import Image

def read_categories(categories_path):
    with open(categories_path, 'r') as file:
        categories = json.load(file)
    # Invertir el mapeo para usar nombres de categorías directamente
    return {v: int(k) for k, v in categories.items()}

def return_indices(start, end, n_frms):
    intervals = np.linspace(start=start, stop=end, num=n_frms + 1).astype(int)
    ranges = []
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1]))
    indices = [(x[0] + x[1]) // 2 for x in ranges]
    return indices

def main(args):
    # Init
    vpath = args['<vpath>']
    outdir = args['<outdir>']
    catfile = args['<catfile>']
    nframes = int(args['--nframes'])

    # Empty annotations
    categories = read_categories(catfile)
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": v, "name": k, "supercategory": "none"} for k, v in categories.items()]
    }

    files = glob.glob(vpath + '/*.jpg')
    indices = return_indices(1, len(files), nframes)
    for idx in indices:
        file = files[idx]
        image_id = int(idx)
        coco['images'].append({
            "id": image_id,
            "file_name": file,
            "width": 320,
            "height": 180
        })
        coco['annotations'].append({
            "id": 0,
            "image_id": image_id,
            "category_id": categories['bag'],   # Default category
            "bbox": [0,0,0,0],
            "area": 320*180,
            "iscrowd": 0 
        })

    # Write the COCO formatted JSON
    with open(outdir, 'w') as out_file:
        json.dump(coco, out_file, indent=4)
    


if __name__ == '__main__':
    args = docopt(__doc__)
    main(args)