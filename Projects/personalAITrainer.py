import mediapipe as mp 
import time
import cv2
import os
import PoseEstimationModule as pm
import numpy as np

pTime=0
detector= pm.poseDetector()
cap= cv2.VideoCapture("E:\Code\Computer Vision\Bicep Workout.mp4")
count=0
dir=0
while True:
    success, img= cap.read()
    img= detector.findPose(img, False)
    lmlist= detector.getPosition(img, False)
    if len(lmlist)!=0:
        #right arm
        rangle= detector.findAngle(img, 12, 14, 16)
        #left arm
        langle =detector.findAngle(img, 11, 13, 15)
        rper= np.interp(rangle,(210,310),(0,100))
        lper= np.interp(langle,(210,310),(0,100))
        rbar = np.interp(rangle, (220, 310), (650, 100))
        lbar = np.interp(langle, (220, 310), (650, 100))

        #if lper ==100 or rper==100:
         #   if dir==0:
          #      count +=0.5
           #     dir=1
       # if lper==0 or rper==0:
        #    if dir==1:
         #       count +=0.5
          #      dir=0 
       # cv2.putText(img, str(count), (50,100), cv2.FONT_HERSHEY_PLAIN,15, (255,0,255),5)
        cv2.rectangle(img, (900, 100), (975, 650), (255,0,255), 3)
        cv2.rectangle(img, (900, int(rbar)), (975, 650), (255,0,255), cv2.FILLED)
        cv2.putText(img, f'{int(rper)}%', (880, 75), cv2.FONT_HERSHEY_PLAIN, 4,
                    (255,0,0), 4)
        
        cv2.rectangle(img, (1100, 100), (1175, 650), (255,0,255), 3)
        cv2.rectangle(img, (1100, int(lbar)), (1175, 650), (255,0,255), cv2.FILLED)
        cv2.putText(img, f'{int(lper)}%', (1080, 75), cv2.FONT_HERSHEY_PLAIN, 4,
                        (255,0,0), 4)

    cTime= time.time()
    fps=1/(cTime-pTime)
    pTime= cTime
    cv2.putText(img, f'FPS: {int(fps)}', (40,50),cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0),2)

    cv2.imshow('Image', img)
    cv2.waitKey(1)
