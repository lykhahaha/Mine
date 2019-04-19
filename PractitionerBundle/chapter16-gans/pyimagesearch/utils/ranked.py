import numpy as np

def rank5_accuracy(preds, labels):
    # initialize rank-1 and rank-5 accuracies
    rank1, rank5 = 0, 0
    # loop over predictions and ground-truth labels
    for p, gt in zip(preds, labels):
        # sort probabilities by their index in descending order so that more confident guesses are at front of list
        p = np.argsort(p)[::-1]# p = [0.11, 0.  , 0.56, 0.33] --> [2, 3, 0, 1]
        # check to see if ground-truth is in top-5 predictions
        if gt in p[:5]:
            rank5+=1
        # check to see if ground-truth is in top-1 predictions
        if gt == p[0]:
            rank1+=1
    # compute final rank-1 and rank-5 accuracies
    rank1/=float(len(preds))
    rank5/=float(len(preds))

    return rank1, rank5