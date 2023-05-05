import cv2
import mediapipe as mp
import time
 
class faceMeshDetector():
    def __init__(self,staticMode=False, redefineLms = False,maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5):
        self.staticMode= staticMode
        self.maxFaces=maxFaces
        self.minDetectionCon=minDetectionCon
        self.minTrackCon=minTrackCon
        self.redefineLms=redefineLms 
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.drawSpecs = self.mpDraw.DrawingSpec(color = (0, 255, 0), thickness = 1, circle_radius = 1)
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces, self.redefineLms, self.minDetectionCon, self.minTrackCon)
        
    
    def findFaceMesh(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        if self.results.multi_face_landmarks:
            faces=[]
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS,
                    self.drawSpecs)
                face=[]
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x,y= int(lm.x*iw), int(lm.y*ih)
                    face.append([id,x,y])
                
        return img , face
          
    
    

def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector= faceMeshDetector()
    while True:
        success, img = cap.read()
        img, face = detector.findFaceMesh(img)
        if len(face)!=0:
            print(face)
        else:
            print(0)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
        3, (255, 0, 0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)



if __name__=='__main__':
    main()