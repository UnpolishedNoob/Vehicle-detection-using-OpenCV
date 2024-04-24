import cv2
import csv
from datetime import datetime


cap=cv2.VideoCapture("opencv_car_video.mp4")
ret,first=cap.read()
ret,second=cap.read()

height,width=first.shape[:2]
count=0

while cap.isOpened():
    diff=cv2.absdiff(first,second)    
    gray=cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(gray,(5,5),0)
    _,thresh=cv2.threshold(blur,20,255,cv2.THRESH_BINARY)
    dilated=cv2.dilate(thresh,None,iterations=4)
    contours,_=cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        (x,y,w,h)=cv2.boundingRect(contour)
        if cv2.contourArea(contour)<1500:
            continue
        cv2.rectangle(first,(x,y),(x+w,y+h),(0,255,0),2)
        c_x=int(x+w/2)
        c_y=int(y+h/2)
        if c_y<(height-147) and c_y>(height-163):
            count+=1
            cv2.circle(first,(c_x,c_y),12,(0,255,255),15)
            break
    cv2.putText(first,"Number of vehicle : "+str(count),(10,25),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,225),3)  
    cv2.imshow("vehicle",first)
    first=second
    ret,second=cap.read()
    if cv2.waitKey(40)==27:
        break
cap.release()
cv2.destroyAllWindows()
c = datetime.now()
current_time = c.strftime('%H:%M:%S')
with open("car.csv","a") as csvfile:
    fieldnames=["Time","Number_of_cars"]
    writer=csv.DictWriter(csvfile,fieldnames=fieldnames)
    writer.writerow({"Time":current_time,"Number_of_cars":str(count)})
