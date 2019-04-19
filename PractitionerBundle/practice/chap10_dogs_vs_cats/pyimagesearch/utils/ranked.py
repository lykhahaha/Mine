import numpy as np

def rank5_accuracy(preds, labels):
    rank_1, rank_5 = 0, 0

    for p, gt in zip(preds, labels):
        p = np.argsort(p)[::-1]

        if gt in p[:5]:
            rank_5+=1
        
        if gt == p[0]:
            rank_1+=1

    rank_1 /= float(len(labels))
    rank_5 /= float(len(labels))

    return rank_1, rank_5