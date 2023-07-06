import cv2

#LOADING THE CASCADE XML FILES WHICH CONTAIN HAAR LIKE FEATURES
#LOADING FACE CASCADE
face_cascade=cv2.CascadeClassifier('face detection\haarcascade_frontalface_default.xml')
#LOADING EYE CASCADE
eye_cascade=cv2.CascadeClassifier('face detection\haarcascade_eye.xml')
#LOADING SMILE CASCADE
smile_cascade=cv2.CascadeClassifier('face detection\haarcascade_smile.xml')

#FUNCTION TO DETECT VARIOUS FEATURES AND DRAW A RECTANGLE OVER THEM
#THIS FUNCTION TAKES TWO INPUTS:
#1. GRAY COLORSPACE FRAME
#2. COLORED FRAMES
#THIS FUCNTION RETURNS COLOURED FRAMES WITH RECTANGLES OVERLAYS THE FEATURES
def detect(gray, frame):
    faces=face_cascade.detectMultiScale(gray,1.3,5) # FACES ARE DETECTED WITH SCALING OF 1.3 AND MIN NEIGHBOUR AS 5 
    #FOR LOOP TO DETECT FACES
    #x: X-COORDINATE OF UPPER LEFT CORNER OF RECTANGLE
    #y: Y-COORDINATE OF UPPER LEFT CORNER OF RECTANGLE
    #w: WIDTH OF RECTANGLE
    #h: HEIGHT OF RECTANGLE
    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2) # WE PAINT A RECTANGLE AROUND THE FACE
        roi_gray=gray[y:y+h,x:x+w] # WE GET THE REGION OF INTEREST IN GRAYSCALE IMAGE
        roi_color=frame[y:y+h,x:x+w] # WE GET THE REGION OF INTEREST IN COLORED IMAGE
        eyes=eye_cascade.detectMultiScale(roi_gray,1.1,15)# EYES ARE DETECTED WITH SCALING OF 1.1 AND MIN NEIGHBOUR AS 15
        for(ex,ey,ew,eh) in eyes: # FOR LOOP TO DETECT EYES
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2) # WE PAINT A RECTANGLE AROUND THE EYES
        smiles=smile_cascade.detectMultiScale(roi_gray,1.7,17) # SMILES ARE DETECTED WITH SCALING OF 1.7 AND MIN NEIGHBOUR AS 17 
        for(sx,sy,sw,sh) in smiles: # FOR LOOP TO DETECT SMILE
            cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(0,0,255),2) # WE PAINT A RECTANGLE AROUND THE SMILING PART
    return frame

#CAPTURING THE VIDEO
video_capture=cv2.VideoCapture(0) # 0 FOR INTERNAL CAM AND 1 FOR EXTERNAL CAM
while(True):
    _,frame=video_capture.read() # READING THE COLORED FRAMES FROM THE VIDEO STREAM
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) # CONVERTING COLORED FRAMES TO GRAY COLORSPACE
    canvas=detect(gray,frame) # detect FUNCTION IS USED AND OVERLAYED FRAMES ARE STORED IN canvas VARIABLE
    cv2.imshow('Video',canvas) # SHOWS THE FRAMES IN A VIDEO LIKE WINDOW
    if(cv2.waitKey(1) & 0xff == ord('q')): # USED TO STOP THE FACE AND FEATURE DETECTION 
        break
video_capture.release() # RELEASES THE WEB CAM
cv2.destroyAllWindows() # CLOSES WINDOWS OPENED BY OPEN CV 