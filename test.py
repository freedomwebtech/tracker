from vidgear.gears import CamGear
import cv2
import imutils

from tracker import *
tracker = EuclideanDistTracker()


stream = CamGear(source='https://www.youtube.com/watch?v=fXdiTW_2iO8', stream_mode = True, logging=True).start() # YouTube Video URL as input

back = cv2.bgsegm.createBackgroundSubtractorMOG();
back2 = cv2.createBackgroundSubtractorMOG2(history=100,varThreshold=50);

while True:
    
      
    frame = stream.read()
    
    frame=cv2.resize(frame,(640,480))
    mask = back2.apply(frame)
    _,mask=cv2.threshold(mask,254,255,cv2.THRESH_BINARY)
    cnts = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    detections=[]
    for c in cnts:
        area = cv2.contourArea(c)
        if area > 400:
#           cv2.drawContours(frame,[c],-1,(0,255,0),3)
#            M=cv2.moments(c)
#            if M["m00"] != 2:
#               cx = int(M["m10"] / M["m00"])
#               cy = int(M["m01"] / M["m00"])
               
               
               x,y,w,h=cv2.boundingRect(c)
#               cx = (x+x+w)//2
#               cy = (y+y+h)//2
        
#               cv2.circle(frame,(cx,cy),2,(0,0,255),2)
               
               detections.append([x,y,w,h])
               
               print(detections)
                             
            
    
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x,y,w,h,id = box_id
        cv2.putText(frame,str(id),(x,y -15),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),2)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow("Output Frame", frame)
#    cv2.waitKey(0)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
stream.stop()