"""
Script to extract frames

Usage: 
    extract_frames_video.py --video=<video> --output=<output> [options]

Options:
    -h --help                             Show this screen.

Arguments:
    <video>                          Path to the folder containing the videos.
    <output>                         Path to the output folder.
"""

import cv2
import os
from docopt import docopt
from tqdm import tqdm
from extract_frames import extract_frames

def main(video, output):
    # Extraer frames
    extract_frames(video, output)

# Uso del script
if __name__ == "__main__":
    args = docopt(__doc__)
    video = args['--video']
    output = args['--output']
    main(video, output)
