from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

# Construct argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-c', '--cascade', required=True, help='path to where the face cascade locates')
ap.add_argument('-m', '--model', required=True, help='path to pre-trained smile detector CNN')
ap.add_argument('-v', '--video', help='(optional) path to the video file')
args = vars(ap.parse_args())

# load the face detector cascade and smile detector CNN
detector = cv2.CascadeClassifier(args['cascade'])

# load model
model = load_model(args['model'])

# if a video path was not supplied, grab the reference to thw webcam
if not args.get('video', False):
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture(args['video'])

# keep looping
while True:
    # grab the current frame
    grabbed, frame = camera.read()
    
    # if we are viewing and did not grab frame, we have reached the ed of the video
    if args.get('video') and not grabbed:
        break

    # resize and convert frame to grayscale, then clone original frame so we can draw on it
    frame = imutils.resize(frame, width=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_clone = frame.copy()

    # detect faces in the input frame
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE, minSize=(30, 30))

    # loop over the face bounding faces
    for f_x, f_y, f_w, f_h in rects:
        # extract ROI of the face from grayscale image, resize it to fixed 28x28 pixels and then prepare the ROI for classification via the CNN
        roi = gray[f_y:f_y+f_h, f_x:f_x+f_w]
        roi = cv2.resize(roi, (28, 28))
        roi = roi.astype('float')/255.
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        
        # determine the label
        pred = model.predict(roi)
        label = 'Smiling' if pred.argmax(axis=1)[0]==1 else 'Not Smiling'

        # display the label and bounding box rectangle on the output frame
        cv2.putText(frame_clone, label, (f_x, f_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(frame_clone, (f_x, f_y), (f_x+f_w, f_y+f_h), (0, 0, 255), 2)
    
    # show our detected faces along with smiling/not smiling labels
    cv2.imshow('Face', frame_clone)

    # stop if q is pressed
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

# cleanup camera and close windows
camera.release()
cv2.destroyAllWindows()