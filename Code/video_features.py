import cv2
import numpy as np

def temporal_diff(frame1, frame2, threshold=50):
    if frame1 == None or frame2 == None:
        return None
    diff = np.abs(frame1.astype('int16') - frame2.astype('int16'))
    diff_t = diff > threshold
    return np.sum(diff_t)


def colorhist_diff(hist1, hist2):
    if hist1 == None or hist2 == None:
        return None
    diff = np.abs(hist1 - hist2)
    return np.sum(diff)




