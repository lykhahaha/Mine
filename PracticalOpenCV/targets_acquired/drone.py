# USAGE
# python drone.py --video FlightDemo.mp4
import argparse
import imutils
import cv2
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
args = vars(ap.parse_args())
 
# load the video
camera = cv2.VideoCapture(args['video'])

# keep looping
while True:
    # grab current frame and initialize status text
    grabbed, frame = camera.read()
    status = 'No Targets'

    # check to see if we  have reached the end of video
    if not grabbed:
        break

    # convert to grayscale, blur it and use canny to detect edges
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(blurred, 50, 150)

    # find contours
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # loop over the contours
    for cnt in cnts:
        # approx the contours by Ramer-Douglas-Peucker algorithm
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.01 * peri, True)

        # check if contour is rectangle
        if len(approx) >= 4 and len(approx) <= 6:
            # compute bonding box of contour
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w/float(h)

            # compute solidity of original contour
            area = cv2.contourArea(cnt)
            hull_area = cv2.contourArea(cv2.convexHull(cnt))
            solidity = area / float(hull_area)

            # enumerate condition to check contour is surely target
            keep_dims = w > 25 and h > 25
            keep_solidity = solidity > 0.9
            keep_aspect_ratio = aspect_ratio >= 0.8 and aspect_ratio <= 1.2

            if keep_dims and keep_solidity and keep_aspect_ratio:
                # draw outline around the target and update status text
                cv2.drawContours(frame, [approx], -1, (0, 0, 255), 4)
                status = 'Target Acquired'

    # draw status text on the frame
    cv2.putText(frame, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # show frame and record if a key is pressed
    cv2.imshow('Frame', frame)
    key = cv2.waitKey(60) & 0xFF

    # if q key is pressed, stop the loop
    if key == ord('q'):
        break

# cleanup camera and close windows
camera.release()
cv2.destroyAllWindows()
