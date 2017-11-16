import numpy as np

def get_iou(box_A, box_B):
    xA1 = box_A[0]
    xA2 = box_A[1]
    yA1 = box_A[2]
    yA2 = box_A[3]
    
    xB1 = box_B[0]
    xB2 = box_B[1]
    yB1 = box_B[2]
    yB2 = box_B[3]

    box_A_area = ( xA2 - xA1 ) * ( yA2 - yA1 )
    box_B_area = ( xB2 - xB1 ) * ( yB2 - yB1 )

    # overlap box
    xx1 = max(xA1, xB1)
    yy1 = max(yA1, yB1)
    xx2 = min(xA2, xB2)
    yy2 = min(yA2, yB2)
    
    # compute the width and height of the bouding box
    w = max(0, xx2 - xx1 + 1)
    h = max(0, yy2 - yy1 + 1)

    overlap_area = w * h
    union_area = box_A_area + box_B_area - overlap_area
    iou = overlap_area / union_area

    return iou

def non_max_supression_with_scores(boxes, scores, threshold):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    idxs = np.argsort(scores)
    suppress = set()

    for i, idx_high_score in enumerate(idxs[::-1]):
        i = len(idxs) - i - 1
        if idx_high_score in suppress:
            continue
        for idx_low_score in idxs[:i]:
            if idx_low_score in suppress:
                continue
            
            iou = get_iou(boxes[idx_high_score], boxes[idx_low_score])
            if (iou > threshold):
                suppress.add(idx_low_score)
                
    pick = set(range(len(boxes))) - suppress 
    return list(pick)        

# Felzenszwalb et al.
def non_max_supression_slow(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    x2 = boxes[:,1]
    y1 = boxes[:,2]
    y2 = boxes[:,3]

    # computer the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes 
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list, add the index
	# value to the list of picked indexes, then initialize
	# the suppression list (i.e. indexes that will be deleted)
	# using the last index
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]

        # loop over all indexes in the indexes list
        for pos in range(last):
            # grab the current index
            j = idxs[pos]
            
            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
	    # for the end of the bounding box
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])

            # compute the width and height of the boudnig box
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)

            # computer the ratio of overlap between the computeed
            # bounding box and the bounding box in the area list
            overlap = float(w * h) / area[j]

            # if there is sufficient overlap,suppress the
            # current bounding box
            if overlap > overlapThresh:
                suppress.append(pos)
        
        # delete all indexes from the index list that are in the
        # suppression list
        idxs = np.delete(idxs, suppress)

    # return only the bounding boxes that were picked
    return boxes[pick]

def non_max_supression_fast():
    pass
