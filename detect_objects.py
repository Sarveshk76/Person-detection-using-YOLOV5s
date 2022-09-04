from heapq import nsmallest
import cv2

import torch

# Model

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model.classes = [0] # only person
# Image

img = cv2.imread('students.jpg')

# Inference

results = model(img)  # includes NMS
# Results

  # print results to screen
results.print() 
persons = results.pandas().xyxy[0]
print("No. of Persons :",len(persons[persons==True]))
# results.save()
