import mediapipe as mp 
import time
import cv2
import os
import handTrackingModule as htm
import numpy as np
import autopy 

wCam, hCam = 640, 480
frameR = 100 # Frame Reduction
smoothening = 7

pTime=0
detector= htm.handDetector(maxHands=1)
cap= cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
xp, yp = 0, 0
while True:
    success, img= cap.read()
    img= detector.findHands(img, False)
    lmList, bbox= detector.findPosition(img)
    if len(lmList):
        x1,y1= lmlist[8][1:]
        x2,y2=lmlist[12][1:]
        print(x1,y1,x2,y2)
    
    
    
    cTime= time.time()
    fps=1/(cTime-pTime)
    pTime= cTime
    cv2.putText(img, f'FPS: {int(fps)}', (40,50),cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0),2)
    cv2.imshow('Image', img)
    cv2.waitKey(1)