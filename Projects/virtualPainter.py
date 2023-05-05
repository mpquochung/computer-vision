import mediapipe as mp 
import time
import cv2
import os
import handTrackingModule as htm
import numpy as np
brushThickness = 25
eraserThickness = 100
folderPath= 'header'
mylist= os.listdir(folderPath)
overlaylist= []
for imPath in mylist:
    image= cv2.imread(f'{folderPath}/{imPath}')
    overlaylist.append(image)
header= overlaylist[0]
drawColor=(255,0,255)
pTime=0
detector= htm.handDetector()
cap= cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)
while True:
    success, img= cap.read()
    img= cv2.flip(img,1)
    img= detector.findHands(img, False)
    lmlist= detector.findPosition(img, draw=False)
    if len(lmlist)!=0:
        xp,yp=0,0
        x1,y1= lmlist[8][1:]
        x2,y2= lmlist[12][1:]
    fingers= detector.fingersUp()
    if fingers[1] and fingers[2]:
        if y1<125:
            if 250<x1<450:
                header= overlaylist[0]
                drawColor=(0,255,0)
            elif 550<x1<750:
                header= overlaylist[1]
                drawColor=(255,0,0)
            elif 820<x1<950:
                header= overlaylist[2]
                drawColor=(203,192,255)
            elif 1050<x1<1200:
                header= overlaylist[3]
                drawColor=(0,0,0)
        cv2.rectangle(img,(x1,y1-25),(x2,y2+25),drawColor, cv2.FILLED )

    if fingers[1] and fingers[2]== False:
        cv2.circle(img,(x1,y1),8, drawColor, cv2.FILLED )
        if xp==0 and yp==0:
            xp, yp=x1,y1
        if drawColor==(0,0,0):
            cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
            cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
        else:
            cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
            cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
        xp,yp=x1,y1
    imgGray=cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv=cv2.threshold(imgGray,  50,255, cv2.THRESH_BINARY_INV)
    imgInv= cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img= cv2.bitwise_and(img,imgInv)
    img= cv2.bitwise_or(img, imgCanvas)

    
    
    img[0:125, 0:1280] = header
    cTime= time.time()
    fps=1/(cTime-pTime)
    pTime= cTime
    cv2.putText(img, f'FPS: {int(fps)}', (40,50),cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0),2)
    cv2.imshow('Image', img)
    cv2.imshow('Canvas', imgCanvas)
    cv2.waitKey(1)
