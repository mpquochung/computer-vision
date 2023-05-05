import mediapipe as mp 
import time
import cv2
import os
import handTrackingModule as htm

wCam, hCam= 640, 480
cap= cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)
folderPath  = 'hands'
myList= os.listdir(folderPath)
overlayList=[]
for imPath in myList:
    image= cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

detector = htm.handDetector(detectionCon=0.75)
pTime=0
tipIDs= [4,8,12,16,20]
while True:
    success, img= cap.read()
    img = detector.findHands(img)
    lmlist= detector.findPosition(img, draw=False)
    if len(lmlist)!=0:
        fingers =[]
        #thumb
        if lmlist[tipIDs[0]][1]>(lmlist[tipIDs[0]-1][1]+1.5):
            fingers.append(1)
        else:
            fingers.append(0)
        #4 fingers
        for id in range(1,5):
            if lmlist[tipIDs[id]][2]<lmlist[tipIDs[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        totalFingers=fingers.count(1)
        print(totalFingers) 
        h,w,c= overlayList[totalFingers].shape
        img[0:h,0:w]=overlayList[totalFingers]
        cv2.putText(img, str(totalFingers), (580, 430), cv2.FONT_HERSHEY_PLAIN,
                    5, (255, 0, 0), 10)
    cTime= time.time()
    fps=1/(cTime-pTime)
    pTime= cTime
    cv2.putText(img, f'FPS: {int(fps)}', (40,50),cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0),2)
    cv2.imshow("Image",img)
    cv2.waitKey(1)

    
