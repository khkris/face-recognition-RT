import numpy as np
import cv2
import pickle
from FR_utils import *

#=======================

print("Press Q to Add to Database when the web-cam opens.")
print("Press any key to open web-cam")
input()

cap = cv2.VideoCapture(0)
haar_face_cascade = cv2.CascadeClassifier('C:/Users/KIIT/AppData/Local/Programs/Python/Python36/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.flip( frame, 1)
    faces = haar_face_cascade.detectMultiScale(frame, 1.3, 5)

    for (x,y,w,h) in faces:

            if w > 80:
                frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(100,100,100),2) #draw rectangle to main image

                detected_face = frame[int(y):int(y+h), int(x):int(x+w)] #crop detected face
                detected_face = cv2.resize(detected_face, (224, 224)) #resize to 224x224

        # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Release capture after all operations
cap.release()
cv2.destroyAllWindows()

# Pre-process image of the detected face
img = img_to_array(detected_face)
img = np.expand_dims(img, axis=0)
img = preprocess_input(img)

FRmodel = FaceModelKK()
print("Enter the name of the individual")
name = input()

# Call the Database and store the encoding 
pickle_out = open("DB.pickle","wb")
add_to_database(name, img, FRmodel)

pickle.dump(Database, pickle_out)
pickle_out.close()

print("Press any key to exit")
input()
