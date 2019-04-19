import numpy as np
import imutils
from keras.applications import imagenet_utils

def sliding_window(image, ws, step):
    for y in range(0, image.shape[0] - ws[1], step):
        for x in range(0, image.shape[1] - ws[0], step):
            yield x, y, image[y:y + ws[1], x:x + ws[0]]

def image_pyramid(image, scale=1.5, min_size=(224, 224)):
    while True:
        w = int(image.shape[1] / scale)

        image = imutils.resize(image, width=w)

        if image.shape[0] < min_size[1] or image.shape[1] < min_size[0]:
            break
        
        yield image

def classify_batch(model, batch_ROIs, batch_locs, labels, min_prob=0.5, top=10, dims=(224, 224)):
    preds = model.predict(batch_ROIs)
    P = imagenet_utils.decode_predictions(preds, top=top)

    for i in range(len(P)):
        for _, label, prob in P[i]:
            if prob > min_prob:
                p_x, p_y = batch_locs[i]
                box = (p_x, p_y, p_x + dims[1], p_y + dims[0])

                l = labels.get(label, [])
                l.append((box, prob))
                labels[label] = l

    return labels

def non_max_suppression(boxes, overlap_thresh, prob=None):
    if boxes.dtype.kind == 'i':
        boxes = boxes.astype('float')

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = y2

    if prob is not None:
        idxs = prob
    pick = []
    idxs = np.argsort(idxs)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.minimum(0, yy2 - yy1 + 1)

        overlap = w * j / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate([last], np.where(overlap < overlap_thresh)[0]))
    
    return boxes[pick].astype('int')ls