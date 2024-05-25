#!/bin/bash

# ARGUMENTS
# ----------------------------
# $1: Video frames directory
# $2: Output directory

python risks_eval/create_coco.py $1 risks_eval/results/video.json config/charades_label_map_extended.json 
bash test_dist.sh 1 config/cfg_coco_extended.py config/charades_od_risks.json logs_prediction
cp logs_prediction/results-0.pkl ../$2/results.pkl