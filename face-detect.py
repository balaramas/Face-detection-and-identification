import numpy as np
import cv2
import io


face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_alt2.xml')
recognizer =cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")



cap=cv2.VideoCapture(0)

while(True):
    
    ret,frame = cap.read()
    gray =cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3)
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w] 
        roi_color = frame[y:y+h, x:x+w]

        id_, conf =recognizer.predict(roi_gray)
        if conf >=50and conf <=90:
            print(id_)
            str=""
            
            if id_==0:
                str="kit harington"
            if id_==1:
                str="narendra modi"
            if id_==2:
                str="christiano ronaldo"
            if id_==3:
                str="barrack obama"
            if id_==4:
                str="emelia clarke"    
            
                
                
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = str
            color = (255, 255, 90)
            stroke=2
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

        
        img_item ="my-image.png"

        cv2.imwrite(img_item,roi_gray)
        

        color = (0,255,255)
        stroke =4
        end_cord_x=x+w
        end_cord_y=y+h
        cv2.rectangle(frame, (x,y),(end_cord_x,end_cord_y), color,stroke)

    
    #show frame after detection
    cv2.imshow('preview',frame)

    #repeat reading frames
    rval, frame = cap.read()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
