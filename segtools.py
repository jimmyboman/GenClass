import numpy as np
import matplotlib.pyplot as plt

##### Function: imgseg
# Takes a numpy array containing an image (img), and a numpy array
# containing a mask of anomalies in img (mask). Segments the image
# into sections of (height,width) and returns one numpy array 
# containing the image segments, and one array of the same length 
# that has a value of 1 if there is an anomaly in the image segment
# at that index, and zero otherwise.

def imgseg(img, mask, height, width, channels, stride):
     
    im_h, im_w = img.shape[:2]

    x = range(0,im_w,stride)
    y = range(0,im_h,stride)

    imgparts = np.zeros((len(x)*len(y), height, width, channels))
    contains_anomaly = np.zeros(len(x)*len(y))
    
    counter = 0
    for row in y[:-2]:
        for col in x[:-2]:
            imgpart = img[row:row+height,col:col+width]
            maskpart = mask[row:row+height,col:col+width]
            try:
                imgparts[counter] = imgpart
            except:
                break
            if np.sum(maskpart) > 0:
                contains_anomaly[counter] = 1
            counter += 1
    return imgparts, contains_anomaly

