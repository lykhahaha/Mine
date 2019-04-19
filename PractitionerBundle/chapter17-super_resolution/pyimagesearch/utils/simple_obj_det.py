from keras.applications import imagenet_utils
import imutils
import numpy as np

def sliding_window(image, step, ws):
    # slide a window across image
    for y in range(0, image.shape[0] - ws[1], step):
        for x in range(0, image.shape[1] - ws[0], step):
            # yield current window
            yield x, y, image[y:y + ws[1], x:x + ws[0]]

def image_pyramid(image, scale=1.5, min_size=(224, 224)):
    # yield original image
    yield image

    # keep looping over the image pyramid
    while True:
        # compute dimensions of the next image in pyramid
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)

        # if resized image is smaller then supplied minimum size, then stop constructing pyramid
        if image.shape[0] < min_size[1] or image.shape[1] < min_size[0]:
            break
        
        # yield next image in pyramid
        yield image

def classify_batch(model, batch_ROIs, batch_locs, labels, min_prob=0.5, top=10, dims=(224, 224)):
    # pass our batch ROIs through network and decode the predictions
    preds = model.predict(batch_ROIs)
    P = imagenet_utils.decode_predictions(preds, top=top) 

    # loop over decoed predictions
    for i in range(len(P)):
        for _, label, prob in P[i]:
            # filter out weak detections by ensuring the predicted prob > min prob
            if prob > min_prob:
                # grab the coords of sliding window for prediction and construct bounding box
                p_x, p_y = batch_locs[i]
                box = (p_x, p_y, p_x + dims[0], p_y + dims[1])

                # grab the list of predictions for the label and add bounding box + prob to list
                L = labels.get(label, [])
                L.append((box, prob))
                labels[label] = L

    # return labels dictionary
    return labels

def non_max_suppression(boxes, overlap_thresh):
    '''
        non max suppression: delete all boxes which have large overlap with largest prob box. First, argsort probability of boxes, if prob does not have, sort by y2 - end_y of boxes, find all IoU of all boxes with largest prob box, If IoU > threshold, delete them, loop until no boxes
        boxes: (start_x, start_y, end_x, end_y)
        overlap threshold normally fall in the range 0.3-0.5
    '''
    # if there si no box, return an empty list
    if len(boxes) == 0:
        return []

    # change to float if boxes is integer
    if boxes.dtype.kind == 'i':
        boxes = boxes.astype('float')
    
    # initialize list of picked bounding boxes
    pick = []

    # grab coords of bounding boxes
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    
    # compute area of bounding boxes
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    # to compute the overlap ratio of other bounding boxes later in this function, need to sort probability or bounding boxes by bottom-right y-coords of bounding box
    idxs = np.argsort(y2)

    # looping all idxs
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        # add last index in indexes list to list of picked indexes
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute width and height of bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute ratio of overlap between computed bounding box and bounding box in area list
        overlap = w * h / area[idxs[:last]]

        # delete all indexes from indexes list that are in suppression list
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))

    return boxes[pick].astype('int')