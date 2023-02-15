import numpy as np
import cv2
import torch
import glob as glob
import os
import time
from model import create_model # Call model
from config import (
    NUM_CLASSES, DEVICE, CLASSES
)
# Ma trận nội tham số Camera
mtx = np.array([[681.51971054 , 0. , 636.14501772], [0. , 679.09038203, 347.70896819] , [0., 0., 1.]])

# Ma trận biến dạng
dist = np.array([[ 0.13836356, -0.19056179,  0.00093916, -0.00244325, -0.02768758]])

# define a video capture object
vid = cv2.VideoCapture(1)

# Check whether user selected camera is opened successfully
if not (vid.isOpened()):
  print("Could not open video device") 

# this will help us create a different color for each class
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load the best model and trained weights
model = create_model(num_classes=NUM_CLASSES)
checkpoint = torch.load('outputs/best_model.pth', map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE).eval()

detection_threshold = 0.8 # ngưỡng 

x = 0 # trục x
y = 0 # trục y

while(True):
      
    # Capture the video frame
    ret, frame = vid.read()

    orig_image = frame.copy()  
    
    ###################################### Tính tọa độ gốc robot ########################################
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Blur using 3 * 3 kernel.
    gray_blurred = cv2.blur(gray, (3, 3))

    # Apply Hough transform on the blurred image.
    detected_circles = cv2.HoughCircles(gray_blurred,
				cv2.HOUGH_GRADIENT, 1, 20, param1 = 50,
			param2 = 30, minRadius = 1, maxRadius = 40)

    # Nếu phát hiện được đường tròn
    if detected_circles is not None: 
        detected_circles = np.uint16(np.around(detected_circles))
        for pt in detected_circles[0, :]: # Lấy hết các cột hàng đầu tiên
            x,y,r = pt[0], pt[1], pt[2]   # tọa độ tâm, bán kính 
            print("Tọa độ gốc :", x, y)
            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
            cv2.circle(frame, (x, y), 1, (0, 0, 255), 3)
            cv2.imshow("Detected Circle", frame)
    ################################## Tính tọa độ gốc Robot, 56 pixel mỗi ô vuông tương đương ...(mm) #################################################

    
    ################################### Detect and Classify ##################################################
    # BGR to RGB
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)

    # make the pixel range between 0 and 1
    image /= 255.0

    # bring color channels to front
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    
    # convert to tensor
    image = torch.tensor(image, dtype=torch.float).cpu() ######### nếu dùng GPU sửa thành ->cuda() ###############

    # add batch dimension
    image = torch.unsqueeze(image, 0)
    with torch.no_grad():
        outputs = model(image.to(DEVICE))

    # load all detection to CPU for further operations
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]


    # carry further only if there are detected boxes
    if len(outputs[0]['boxes']) != 0:
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        # filter out boxes according to `detection_threshold`
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        draw_boxes = boxes.copy()
        # get all the predicited class names
        pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
        
        # draw the bounding boxes and write the class name on top of it
        for j, box in enumerate(draw_boxes):
            class_name = pred_classes[j]           #class_name : Tên class dự đoán
            print(box);                            #(box[0],box[1]): tọa độ góc trên của boundingbox, (box[2],box[3]): tọa độ góc dưới của boundingbox
            color = COLORS[CLASSES.index(class_name)]
            a = int ((int((box[2]-box[0])/2)+box[0] - x)/56 *23) # tọa độ theo trục x của tâm boundingbox so với hệ robot
            b = int ((int((box[3]-box[1])/2)+box[1] - y)/56 *23) # tọa độ theo trục y của tâm boundingbox so với hệ robot
            cv2.rectangle(orig_image,
                        (int(box[0]), int(box[1])),
                        (int(box[2]), int(box[3])),
                        color, 2)
            cv2.putText(orig_image, class_name + f"({a}," + f"{b})", 
                        (int(box[0]), int(box[1]-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 
                        2, lineType=cv2.LINE_AA)
            cv2.putText(orig_image,"*",(abs(int((box[2]-box[0])/2))+box[0], abs(int((box[3]-box[1])/2))+box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))

        cv2.imshow('Prediction', orig_image)
        cv2.waitKey(1)
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
