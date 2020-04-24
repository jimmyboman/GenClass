import numpy as np


##### Function: imgseg
#
# Takes a numpy array containing an image (img), and a numpy array
# containing a mask of anomalies in img (mask). Segments the image
# into sections of (height,width) and returns one numpy array 
# containing the image segments, one array of the same length 
# that has a value of 1 if there is an anomaly in the image segment
# at that index and zero otherwise, and the amount of rows 
# and columns in the image array.
#
# Input:
#   - img: Image to be segmented, numpy array
#   - mask: A mask of the anomalies in the image. Numpy array
#           that is zero everywhere except the pixels where
#           the image is anomalous. Can be zero-array if one is 
#           not interested in labelling the segments as anomalous
#           or not, such as when classifying a new image
#   - height: Height of the image segments, integer
#   - width: Width of the image segments, integer
#   - channels: Number of channels in the image, integer
#   - stride: Distance to move between segments. Determines
#             how much the segments overlap.
#
# Output:
#   - imgparts: Numpy array containing the image segments. Has dimensions
#               (xtot,ytot,height,width,channels)
#   - contains anomaly: Labels for each segment, 1 for anomaly and 0 otherwise.
#                       Numpy array of dimensions (xtot,ytot)
#   - xtot: Number of columns of segments, integer
#   - ytot: Number of rows of segments, integer

def imgseg(img, mask, height, width, channels, stride):
    
    # Get image dimensions
    im_h, im_w = img.shape[:2]

    # Determine the locations of the image segments
    x = range(0,im_w-width,stride)
    y = range(0,im_h-height,stride)

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
            # end of each row/col. If the last column is reached
            # the segmentis discarded and we move to the next 
            # row, if there is one. If the last row has been 
            # reached, the segmentation is done.
            imgparts[xcounter][ycounter] = imgpart
            
            
            # If the mask is non-zero in the current segment,
            # it has an anomaly
            if np.sum(maskpart) > 0:
                contains_anomaly[xcounter][ycounter] = 1
            
            # Count up the number of columns, store as total if it is higher
            # This ensures that columns will only be counted on the first row
            xcounter += 1
            if xcounter > xtot:
                xtot = xcounter
                
        # Count up the number of rows, reset the column counter        
        ycounter += 1
        xcounter = 0
    ytot = ycounter
    return imgparts, contains_anomaly, xtot, ytot
    
##### Function: anom_scores
#
# Using an already trained classifier, scans an image for anomalies and
# returns a map of pixel-wise anomaly scores. The scan is done in segments,
# with overlap based on the stride parameter. Higher anomaly score indicates
# higher probability that a given pixel is anomalous.
#
# Input:
#   - model_type: 'cnn' or 'autoencoder', which type of classifier to use
#   - model: Trained model to use for classification
#   - image: The image to be tested for anomalies, numpy array
#   - seg_w: Width of the image segments, integer
#   - seg_h: Height of the image segments, integer
#   - stride: Distance to move between segments, integer
#
# Output:
#   - anomaly_map: Pixel-wise anomaly scores, numpy array
#                  with same height and width as input image
#                  but will always have one channel

    
def anom_scores(model_type, model, image, seg_w, seg_h, channels, stride):
    
    # Create empty mask to pass to imgseg because we do not yet know
    # where the anomalies are
    mask = np.zeros_like(image)
    # Run imgseg on image to divide it into segments, reshape into 4 dimensions
    segments, _, rows, cols = imgseg(image, mask, seg_w, seg_h, channels, stride)
    segments = segments.reshape(-1,seg_w,seg_h,channels)
    
    # Create anomaly map with same height and width as input image
    anomaly_map = np.zeros(image.shape[:2])
    
    # Use the model to make a prediction on each image
    preds = model.predict(segments)
    
    for i in range(rows):
        for j in range(cols):
            # Use binary prediction for CNN type, and reconstruction error for autoencoder, as anomaly score
            if model_type == 'cnn':
                # CNN returns negative number for non-anomaly, positive for anomaly. 
                # This is converted into a binary score, 1 for anomaly and 0 for non-anomaly.
                anomaly_map[j*stride:j*stride+seg_h,i*stride:i*stride+seg_w] += preds[i*cols+j] > 0
            elif model_type == 'autoencoder':
                anomaly_map[j*stride:j*stride+seg_h,i*stride:i*stride+seg_w] += np.sum((preds[i*cols+j] - segments[i*cols+j])**2)
                
    # Return the map
    return anomaly_map
    
def anom_scores_partial(parts, model_type, model, image, seg_w, seg_h, channels, stride, row_print=False):

    w, h = image.shape[:2]
    anomaly_map = np.zeros((w,h))

    for i in range(parts):
        for j in range(parts):
        
            print(f'x = {i*w//parts}:{w//parts*(i+1)}, y = {j*h//parts}:{h//parts*(j+1)}')
            
            anomaly_map[i*w//parts:w//parts*(i+1),
                j*h//parts:h//parts*(j+1)] = anom_scores(model_type=model_type, 
                                                            model=model,
                                                            image=image[i*w//parts:w//parts*(i+1),
                                                                   j*h//parts:h//parts*(j+1)],
                                                            seg_w=seg_w, 
                                                            seg_h=seg_h, 
                                                            channels=channels, 
                                                            stride=stride
                                                            )
                                                            
    return anomaly_map