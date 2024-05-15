#!/bin/bash

# ARGUMENTS
# --------------------------------------------
# $1: Path to the input video
# $2: Path to the output directory

# check video exists
if [ -f "$1" ]; then
    echo "Video check correct."
else
    echo "The video file does not exist. Stopping the program."
    exit 1
fi
# create output dir if not exist
if [ ! -d "$2" ]; then
    mkdir -p "$2"
fi
if [ ! -d "$2/frames" ]; then
    mkdir -p "$2/frames"
fi

# extract frames from video
echo "Extracting frames from video..."
python feature_extraction/extract_frames_video.py --video $1 --output $2/frames/

# execute grounding dino
echo "Executing grounding dino..."
python opengdino/risks_eval/create_coco.py $2/frames opengdino/risks_eval/results/video.json opengdino/config/charades_label_map.json 
cd opengdino && bash test_dist.sh 1 config/cfg_coco.py config/charades_od_risks.json logs_prediction
cd ..
cp opengdino/logs_prediction/results-0.pkl $2/results.pkl

# execute sevila questions
echo "Executing sevila questions..."
python sevila_risks.py $1 $2/sevila_answers.csv

# generate final results
echo "Generating final results..."
python gendesc_and_classify.py --objects_file $2/results.pkl --answer_file $2/sevila_answers.csv --label_map opengdino/config/charades_label_map.json

# remove temporal files
rm -r $2/frames