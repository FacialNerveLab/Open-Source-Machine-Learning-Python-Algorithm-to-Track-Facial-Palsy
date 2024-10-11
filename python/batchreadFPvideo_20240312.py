#readFPvideo

import cv2
import numpy as np
import math
import dlib
import csv
from datetime import datetime
import os
import glob
from os.path import normpath, basename
import shutil

# Load the detector
detector = dlib.get_frontal_face_detector()

# Load the predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#batch process videos in a file with all the SCBS videos
#**use if going to batch process:  directory = "C:\\Users\\jjgre\\Batch_Video\\Synkinetic\\missing"
directory = "C:\\Users\\jjgre\\toanalyze3"
videos = []

for root, subdirectories, files in os.walk(directory):
    for file in files:
        file_location = os.path.join(root,file)
        if file.endswith(".MP4"): #will mess up if MP4 vs mp4
            count = 0
            filenames = glob.glob(file_location)
            videos.append(filenames)

#to convert videos list data to strings so can open
index = 0
while index < len(videos):
    SCBSvideo = ' '.join(videos[index])#converts element of list to a string
    SCBSname = basename(normpath(SCBSvideo))
    SCBSname = SCBSname.partition(".")[0]

    cap = cv2.VideoCapture(SCBSvideo)   # capturing the video from the given path
    # #check if opened this successfully
    if cap.isOpened() == False:
        print("Unable to read video")

    #save processed video
    out = cv2.VideoWriter(SCBSname+'_pythonlandmarks.mp4', 0x7634706d, 30.0, (720,1280))
    #if video is frame rate 30 720 x 480     
    #out = cv2.VideoWriter(filename+'_pythonlandmarks.mp4', 0x7634706d, 30.0, (720,480))

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
    filename = datetime.now().strftime(SCBSname+'_pythonlandmarks_%Y%m%d_%H%M.csv')
    with open(filename, 'w', newline = '') as csvfile: 
                
        # creating a csv writer object 
        csvwriter = csv.writer(csvfile)
        # writing the fields 
        csvwriter.writerow(fields) 
        
    while cap.isOpened():
        ret, frame = cap.read()
        #if frame is read correctly ret is true, will break if not working
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

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
                cv2.circle(img=frame, center=(x, y), radius=3, color=(0, 255, 0), thickness=-1)

            
                #output landmarks to CSV
                with open(filename, 'a', newline = '') as csvfile: 
                    
                    # creating a csv writer object 
                    csvwriter = csv.writer(csvfile)
         
                    # writing the data rows 
                    csvwriter.writerow(output)
                    csvfile.close()

        # show the image
        #cv2.imshow(winname="Face", mat=frame)
        #write processed frames to mp4
        out.write(frame)
        
        # Exit when escape is pressed
        if cv2.waitKey(delay=1) == 27:
            break
      
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    index +=1

print("done!")
