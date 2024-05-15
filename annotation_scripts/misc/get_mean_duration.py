import os
import cv2

# List of video file paths
with open('data/charades_filtered_videos_gpt3.txt', 'r') as f:
  video_files = f.readlines()
video_files = [x.strip() for x in video_files]

# Initialize list to store durations
durations = []

# Iterate through each video file
for video_file in video_files:
    # Open the video file
    vpath = os.path.join('/charades/', "{}.mp4".format(video_file))
    cap = cv2.VideoCapture(vpath)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_file}")
        continue
    
    # Get total number of frames and frames per second (fps)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate duration (in seconds)
    duration = total_frames / fps
    
    # Append duration to list
    durations.append(duration)
    
    # Release the video capture object
    cap.release()

# Calculate mean duration
mean_duration = sum(durations) / len(durations)

print("Mean Duration (seconds):", mean_duration)
