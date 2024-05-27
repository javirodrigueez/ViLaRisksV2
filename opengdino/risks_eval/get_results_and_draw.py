import cv2
import torch
import json

# Load the image
#image_path = '/charades/frames/5VUT9/5VUT9-000857.jpg'
image_path = '../ztest/frames/frame_0283.jpg'
image = cv2.imread(image_path)

# Load the label map
with open('./config/charades_label_map_extended.json', 'r') as file:
    label_map = json.load(file)

# Load the model's output object with bounding box data
model_path = 'logs_prediction/results-0.pkl'
object = torch.load(model_path)
# Iterate through each bounding box in the object
for bbox_data in object['res_info'][10]:
    # bbox_data format expected: [x1, y1, x2, y2, confidence]
    x1, y1, x2, y2, confidence, label, frame_idx = bbox_data
    label_str = label_map[str(int(label))]
    if '_' in label_str:
        label_str = label_str.split('_')[0]
    # Check if the confidence is greater than 0.5
    #and (label_str == 'person' or label_str == 'laptop' or label_str == 'cup' or label_str == 'closet' or label_str == 'kettle')
    if confidence > 0.25 and (label_str == 'person' or label_str == 'groceries'):
        print(f'Label: {label_str}, Confidence: {confidence}')
        # Draw the bounding box on the image
        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0
        if x2 > 224:
            x2 = 223
        if y2 > 224:
            y2 = 223
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
        # draw the label and confidence on the image
        # label_str = label_map[str(int(label))]
        # if '_' in label_str:
        #     label_str = label_str.split('_')[0]
        # text = f'{label_str}'
        #cv2.putText(image, text, (int(x1)+2, int(y1)+12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# Optionally, save the image with the bounding boxes to a file
cv2.imwrite('risks_eval/bbox.jpg', image)
