import cv2
import numpy as np
from matplotlib import pyplot as pl

stream_cam = cv2.VideoCapture(0)


#video streaming
while True:
    # `success` is a boolean and `frame` contains the next video frame
    success, frame = stream_cam.read()


    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = cv2.CascadeClassifier('face_detect_model.xml')
    results = faces.detectMultiScale(frame_gray, scaleFactor= 1.3, minNeighbors= 2)
    
    for(x, y, w, h) in results:
        cv2.rectangle(frame, (x,y), (x + w, y + h), (100,20,230), thickness=2)


    cv2.imshow("frame", frame)
    if cv2.waitKey(10) == ord('q'):
        break
