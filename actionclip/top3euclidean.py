"""
Script to get top3 AR score using Euclidean distance from top5 ActionCLIP.

Usage:
  top3euclidean.py <objects> <top5> [options]

Arguments:
    <objects>               Path to the file containing the objects.
    <top5>                  Path to the file containing the top5.

Options:
    -h --help                   Show this screen.
    --label_map=<label_map>     Path to the label map. [default: ../opengdino/config/charades_label_map_extended.json]
    --verbs_map=<verbs_map>     Path to the verbs map. [default: charades_files/verbs_map.json]
    --coco=<coco>               Path to the COCO dataset. [default: ../opengdino/config/charades_coco.json]
"""

from docopt import docopt
import gensim.downloader as api
import math
import json
import torch
from tqdm import tqdm

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

def get_main_object(bboxes, label_map):
    # get object with label person
    person_boxes = []
    for bbox in bboxes:
        obj_str = label_map[str(int(bbox[5].item()))]
        if obj_str == 'person':
            person_boxes.append(bbox)
    if person_boxes == []:
        return None
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

def get_main_actions(wv, obj, actions, top_n=3):
    # Calculate similarities
    similarities = []
    for action in actions:
        sim = wv.similarity(obj, action)
        similarities.append((action, sim))
    # Sort the actions based on similarity scores in descending order
    similarities.sort(key=lambda x: x[1], reverse=True)
    # Get the top N actions
    top_actions = similarities[:top_n]
    return [action for action, sim in top_actions]

def get_coco_ids(video, coco):
    ids = []
    for frame in coco['images']:
        if video in frame['file_name']:
            ids.append(frame['id'])
    return ids

def filter_objects(objects, ids):
    return list(filter(lambda x: int(x[6].item()) in ids, objects))

def main(args, wv):
    # Load objects
    objects_file = args['<objects>']
    label_map = args['--label_map']
    verbs_map = args['--verbs_map']
    coco = args['--coco']
    with open(label_map, 'r') as file:
        label_map = json.load(file)
    with open(verbs_map, 'r') as file:
        verbs_map = json.load(file)
    with open(coco, 'r') as file:
        coco = json.load(file)
    all_objects = torch.load(objects_file)
    thresh_obj = [[obj if obj[4] > 0.25 else None for obj in frame] for frame in all_objects['res_info']]
    thresh_obj = [obj for frame in thresh_obj for obj in frame]
    thresh_obj = list(filter(lambda x: x is not None, thresh_obj))
    # Load actions
    with open(args['<top5>'], 'r') as f:
        actions = f.readlines()
    actions = [action.strip() for action in actions]
    correct = 0
    zero_objs = 0
    for a in tqdm(actions):
        # init
        video = a.split(' ')[0].split('.')[0].split('/')[-1]
        top5 = a.split(' ')[2].split(';')
        top3gt = a.split(' ')[1].split(';')
        top3gt_str = [verbs_map[action] for action in top3gt]

        # get associated objects
        ids = get_coco_ids(video, coco)
        objects = filter_objects(thresh_obj, ids)
        if len(objects) == 0:
            zero_objs += 1
            continue
        # get main object
        main_obj = get_main_object(objects, label_map)
        if main_obj is None:
            zero_objs += 1
            continue
        main_obj_str = label_map[str(int(main_obj[5].item()))]
        if '_' in main_obj_str:
            main_obj_str = main_obj_str.split('_')[0]
        # get new top3 actions
        top5_str = [verbs_map[action] for action in top5]
        top3_str = get_main_actions(wv, main_obj_str, top5_str)
        # check
        common_actions = set(top3_str).intersection(set(top3gt_str))
        if len(common_actions) > 0:
            correct += 1
    print(f'Top3 Score: {correct / (len(actions)-zero_objs) * 100}%')



if __name__ == '__main__':
    args = docopt(__doc__)
    wv = api.load('word2vec-google-news-300')
    main(args, wv)