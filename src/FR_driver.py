

import numpy as np
import cv2
import pickle
from FR_utils import *

#=====================

cap = cv2.VideoCapture(0) # Calling web-cam for capturing frames
haar_face_cascade = cv2.CascadeClassifier('C:/Users/KIIT/AppData/Local/Programs/Python/Python36/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
FRmodel = FaceModelKK() # Call the Neural Network Model

# Call in Database for reading
pickle_in = open("DB.pickle", "rb")
Database = pickle.load(pickle_in)

while(True):
    
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.flip( frame, 1)
    
    faces = haar_face_cascade.detectMultiScale(frame, 1.3, 5)

    for (x,y,w,h) in faces:

            if w > 80:
                frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(100,100,100),2) # Draw rectangle to main image

                detected_face = frame[int(y):int(y+h), int(x):int(x+w)] # Crop detected face
                detected_face = cv2.resize(detected_face, (224, 224)) # Resize to 224x224
                
                img = img_to_array(detected_face)
                img = np.expand_dims(img, axis=0)
                img = preprocess_input(img)
                
                dist, identity = recognition(img, Database, FRmodel)
                
                cv2.putText(frame, identity, (int(x+w+15), int(y-12)) ,cv2.FONT_HERSHEY_TRIPLEX, 1, (100,100,100), 2)
                #print(dist)
                
                # Connect the text and face with a line
                cv2.line(frame,(int((x+x+w)/2),y+15),(x+w,y-20),(100,100,100),1)
                
        # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release capture after all operations
cap.release()
cv2.destroyAllWindows()

print("Press any key to exit")
input()

