import numpy as np

##### Function: imgseg
# Takes a numpy array containing an image (img), and a numpy array
# containing a mask of anomalies in img (mask). Segments the image
# into sections of (height,width) and returns one array containing 
# the image segments, and one array of the same length that has a
# value of 1 if there is an anomaly in the image segment at that 
# index, and zero otherwise.

def imgseg(img, mask, height, width):
    
    imgparts = []
    contains_anomaly = []
    
    im_h, im_w = img.shape[:2]

    x = range(0,im_w,width//2)
    y = range(0,im_h,height//2)

    for row in y[:-1]:
        for col in x[:-1]:
            imgpart = img[row:row+height,col:col+width]
            maskpart = mask[row:row+height,col:col+width]
            
            imgparts.append(imgpart)
            if np.sum(maskpart) > 0:
                contains_anomaly.append(1)
            else:
                contains_anomaly.append(0)
    return imgparts, contains_anomaly

