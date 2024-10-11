#readFPvideo

import cv2
import numpy as np
import math
import dlib
import csv
from datetime import datetime

# Load the detector
detector = dlib.get_frontal_face_detector()

# Load the predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

count = 0
#videoFile = "face.jpg"

videoFile = "C:\\Users\\jjgre\\Batch_Video\\Normals\\Normal1.mp4"
cap = cv2.VideoCapture(videoFile)   # capturing the video from the given path
#check if opened this successfully
if cap.isOpened() == False:
    print("Unable to read video")

#save processed video to your directory
out = cv2.VideoWriter('PreopFP_pythonlandmarks.mp4', 0x7634706d, 30.0, (720,1280))
# out = cv2.VideoWriter('Normal10_pythonlandmarks.mp4', 0x7634706d, 60.0, (1080,1920))


#read frame rate, video width and height
frameRate = cap.get(5) #frame rate
print ("frame rate", frameRate)
x=1
    
#read frame rate, video width and height
    #frameRate = cap.get(5) # frame rate
    #print("frame rate: ", frameRate)
w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print("frame width x height: ",w, h)

# CSV column field names 
fields = ['frame', 'Facial Landmark #', 'x', 'y'] 
    
# name of csv file 
#filename = datetime.now().strftime(filename+'_%Y%m%d_%H%M.csv')
filename = datetime.now().strftime('Normal1_pythonlandmarks.csv')
with open(filename, 'w', newline = '') as csvfile: 
                
# creating a csv writer object 
    csvwriter = csv.writer(csvfile)
# writing the fields 
    csvwriter.writerow(fields) 
        
while cap.isOpened():
    ret, frame = cap.read()
    frameID = cap.get(1) #current frame number
    # Convert image into grayscale
    gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)

    # Use detector to find landmarks
    faces = detector(gray)

    for face in faces:
        x1 = face.left()  # left point
        y1 = face.top()  # top point
        x2 = face.right()  # right point
        y2 = face.bottom()  # bottom point

        # Create landmark object
        landmarks = predictor(image=gray, box=face)

        # Loop through all the points
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            output = (frameID, n, x, y)

            # Draw a circle
            cv2.circle(img=frame, center=(x, y), radius=1, color=(0, 255, 0), thickness=-1)

        
            #output landmarks to CSV
            with open(filename, 'a', newline = '') as csvfile: 
                
                # creating a csv writer object 
                csvwriter = csv.writer(csvfile)
     
                # writing the data rows 
                csvwriter.writerow(output)
                csvfile.close()

    # show the image
    cv2.imshow(winname="Face", mat=frame)
    #write processed frames to mp4
    out.write(frame)
    
    # Exit when escape is pressed
    if cv2.waitKey(delay=1) == 27:
        break
  
cap.release()
out.release()
print ("Done!")
cv2.destroyAllWindows()
