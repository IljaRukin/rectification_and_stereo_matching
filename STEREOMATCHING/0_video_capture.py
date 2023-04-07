import numpy as np
import cv2
from time import sleep

cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)

i = 0

while True:
    ret0, frame0 = cap0.read()
    cv2.imshow('frame0',frame0)
    #gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('frame0',gray0)
	
    ret1, frame1 = cap1.read()
    cv2.imshow('frame1',frame1)
    #gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('frame1',gray1)
	
    key = cv2.waitKey(0)
    if key == ord('q'):
        break
    else:
        i+=1
        cv2.imwrite('./images_left/image'+str(i)+'.png',frame0)
        cv2.imwrite('./images_right/image'+str(i)+'.png',frame1)

cap0.release()
cap1.release()
cv2.destroyAllWindows()