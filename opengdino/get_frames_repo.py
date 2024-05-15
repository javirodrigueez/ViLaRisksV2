import cv2
import os

# Define paths
viddataset = '/USERPATH/Charades_v1_480/'  # Path to the directory containing video files
imgdataset = '/USERPATH/Charades_v1_480_images/'  # Path where images will be saved

# Create the directory for images if it doesn't exist
if not os.path.exists(imgdataset):
    os.makedirs(imgdataset)

# List all files in the video directory
files = os.listdir(viddataset)

# Loop through each file in the directory
for file in files:
    if file.endswith(".mp4"):  # Check if the file is a video
        vidname = file[:-4]  # Remove the .mp4 extension to get the video name
        vidpath = os.path.join(viddataset, file)  # Construct the full path to the video file
        
        # Read the video
        cap = cv2.VideoCapture(vidpath)
        
        # Create a directory for the current video frames
        impath = os.path.join(imgdataset, vidname)
        if not os.path.exists(impath):
            os.makedirs(impath)
        
        # Frame counter
        im_id = 1
        
        # Read frames from the video
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # If no frame is returned, the video has ended
            
            # Save the frame as an image
            im_name = "{}.jpg".format(im_id)
            cv2.imwrite(os.path.join(impath, im_name), frame)
            
            im_id += 1
        
        # Release the video capture object
        cap.release()

        print(f"Processed video: {vidname}")
