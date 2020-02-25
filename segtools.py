import numpy as np


##### Function: imgseg
# Takes a numpy array containing an image (img), and a numpy array
# containing a mask of anomalies in img (mask). Segments the image
# into sections of (height,width) and returns one numpy array 
# containing the image segments, one array of the same length 
# that has a value of 1 if there is an anomaly in the image segment
# at that index and zero otherwise, and the amount of rows 
# and columns in the image array.

def imgseg(img, mask, height, width, channels, stride):
    
    # Get image dimensions
    im_h, im_w = img.shape[:2]

    # Determine the locations of the image segments
    x = range(0,im_w,stride)
    y = range(0,im_h,stride)

    # Allocate arrays to store segments and anomaly indicator
    # Segment array has shape (nx, ny, height, width, channels)
    imgparts = np.zeros((len(x), len(y), height, width, channels))
    contains_anomaly = np.zeros((len(x),len(y)))
    
    # Keep track of how many rows and columns of segments there are
    xcounter = 0
    xtot = 0
    ycounter = 0
    ytot = 0
    
    # Loop over rows and columns of segments
    for row in y:
        for col in x:
            # Fetch current segment from image and mask
            imgpart = img[row:row+height,col:col+width]
            maskpart = mask[row:row+height,col:col+width]
            
            # Attempt to store segment in segment array
            # This will fail if the segment is not of size
            # (height, width), which happens when the segment 
            # exceeds the boundary of the image, i.e. at the 
            # end of each row/col. These segments are discarded
            try:
                imgparts[xcounter][ycounter] = imgpart
            except:
                ytot = ycounter
                break
            
            # If the mask is non-zero in the current segment,
            # it has an anomaly
            if np.sum(maskpart) > 0:
                contains_anomaly[xcounter][ycounter] = 1
                
            xcounter += 1
            if xcounter > xtot:
                xtot = xcounter
                
        ycounter += 1
        xcounter = 0

    return imgparts[:xtot,:ytot], contains_anomaly[:xtot,:ytot], xtot, ytot

