import cv2

# Initiate video capture for video file
cap = cv2.VideoCapture('walking.avi')

body_classifier=cv2.CascadeClassifier('haarcascade_fullbody.xml')

# Loop once video is successfully loaded
while True:
    
    # Read first frame
    ret, frame = cap.read()

    #Convert Each Frame into Grayscale
    grey=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Pass frame to our body classifier
    bodies=body_classifier.detectMultiScale(grey, 1.2, 3)
    
    # Extract bounding boxes for any bodies identified
    for(x,y,w,h) in bodies:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)

        cv2.imshow('ccdv', frame)

    if cv2.waitKey(1) == 32: #32 is the Space Key
        break

cap.release()

cv2.destroyAllWindows()
