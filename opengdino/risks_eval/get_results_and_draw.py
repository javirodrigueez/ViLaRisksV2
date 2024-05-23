import cv2
import torch

# Load the image
#image_path = '/charades/frames/5VUT9/5VUT9-000857.jpg'
image_path = '../ztest/frames/frame_0857.jpg'
image = cv2.imread(image_path)

# Load the model's output object with bounding box data
model_path = 'logs_prediction/results-0.pkl'
object = torch.load(model_path)
# Iterate through each bounding box in the object
for bbox_data in object['res_info'][25]:
    # bbox_data format expected: [x1, y1, x2, y2, confidence]
    x1, y1, x2, y2, confidence, label, frame_idx = bbox_data
    
    # Check if the confidence is greater than 0.5
    if confidence > 0.25 and int(label.item())==16:
        # Draw the bounding box on the image
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)

# Optionally, save the image with the bounding boxes to a file
cv2.imwrite('risks_eval/bbox.jpg', image)
