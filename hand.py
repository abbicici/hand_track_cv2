from unittest import result
import cv2
import time
import mediapipe as mp

cap=cv2.VideoCapture(1)
pTime=0
cTime=0

mpHand=mp.solutions.hands
hands=mpHand.Hands()

mpDraw=mp.solutions.drawing_utils

while True:
    success,img=cap.read()
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=hands.process(imgRGB)
    print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img,handLms,mpHand.HAND_CONNECTIONS)
            for id,lm in enumerate(handLms.landmark):
                #print(id,lm)
                h,w,c=img.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                if id==20:
                    cv2.circle(img,(cx,cy),9,(255,0,0),cv2.FILLED)

    
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime

    cv2.putText(img,"fps:"+str(int(fps)),(10,75),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)

    cv2.imshow("img",img)
    cv2.waitKey(1)
