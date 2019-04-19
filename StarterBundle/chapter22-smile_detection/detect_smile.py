# USAGE
# python detect_smile.py --cascade haarcascade_frontalface_default.xml --model output/lenet.hdf5
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

# Construct argument parser and parse an argument
ap = argparse.ArgumentParser()
ap.add_argument('-c', '--cascade', required=True, help='path to cascade file')
ap.add_argument('-m', '--model', required=True, help='path to pretrained weights for smile detection')
ap.add_argument('-v', '--video', help='(optional) path to video file')
args = vars(ap.parse_args())

# load face detector cascade and smile detector CNN
detector = cv2.CascadeClassifier(args['cascade'])
model = load_model(args['model'])

# if video path is not supplied, refer to webcam, otherwise load video
if not args.get('video', False):
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture(args['video'])

# keep looping
while True:
    # get the current frame
    grabbed, frame = camera.read()

    # if we are viewing a video and we did not grab a frame, we reach the end of video
    if args.get('video') and not grabbed:
        break
    
    # resize frame, convert it to grayscale, and clone the original frame so we can draw it later
    frame = imutils.resize(frame, width=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_clone = frame.copy()

    # detect faces in input frame
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    # loop over face
    for fX, fY, fW, fH in rects:
        # extract ROI of face from grayscale image, resize it to (28, 28), the prepare ROI for classification
        roi = gray[fY:fY+fH, fX:fX+fW]
        roi = cv2.resize(roi, (28, 28))
        roi = roi.astype('float')/255
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # get prob of each class
        prediction = model.predict(roi)[0]
        label = 'Smiling' if prediction.argmax()==1 else 'Not Smiling'
        # display the label and bounding box rectangle on the output frame
        cv2.putText(frame_clone, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(frame_clone, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)

    # show our detected faces along with label
    cv2.imshow('Face', frame_clone)

    # if 'q' is pressed, stop loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# cleanup camera and close any open windows
camera.release()
cv2.destroyAllWindows()