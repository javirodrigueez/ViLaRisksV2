import cv2
import numpy as np
import os

# List of video file paths
with open('data/charades_filtered_videos_gpt3.txt', 'r') as f:
  video_files = f.readlines()
video_files = [x.strip() for x in video_files]

# Initialize lists to store heights and widths
heights = []
widths = []

# Iterate through each video file
for video_file in video_files:
    vpath = os.path.join('/charades/', "{}.mp4".format(video_file))
    # Open the video file
    cap = cv2.VideoCapture(vpath)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_file}")
        continue
    
    # Get height and width of the video
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    # Append height and width to lists
    heights.append(height)
    widths.append(width)
    
    # Release the video capture object
    cap.release()

# Calculate median height and width
median_height = np.median(heights)
median_width = np.median(widths)

print("Median Height:", median_height)
print("Median Width:", median_width)
