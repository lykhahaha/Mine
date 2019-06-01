# pad and resize our input images to a fixed size without distorting their aspect ratio
import imutils
import cv2

def preprocess(image, width, height):
    # grab dimensions of image, the initialize padding values
    h, w = image.shape[:2]

    # if width is greater than height, resize along width
    if w > h:
        image = imutils.resize(image, width=w) # imutils.resize keep ratio aspect rather than cv2.resize
    # otherwise resize by height
    else:
        image = imutils.resize(image, height=h)
    
    # determine padding values for width and height to obtain target dimensions
    pad_w = (width - image.shape[1])//2
    pad_h = (height - image.shape[0])//2

    # pad image then applly one cv2.resize to handle round issues
    image = cv2.copyMakeBorder(image, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (width, height))

    return image