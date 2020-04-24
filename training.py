import numpy as np
from os import listdir
from segtools_fast import imgseg
from matplotlib.pyplot import imread
from tensorflow.keras.preprocessing.image import ImageDataGenerator

##### Function: training_loader
#
# Loads all the images in the 'impath' directory, as well as the masks
# in the 'maskpath' directory, to create training data by segmenting the
# image into seg_size-by-seg_size squares. Masks should be the same size
# as the images, with pixel value 1 where the image contains an anomaly,
# and 0 everywhere else. Returns one array containing all training images
# and one array with the ground truth value for each training image.
#
# Input:
#   - impath: Path to directory containing training images.
#   - maskpath: Path to directory containing training masks.
#   - seg_size: Desired segment size in pixels, integer.
#   - channels: Number of channels in image, integer.
#
# Output:
#   - x_train: Numpy array of shape (n,seg_size,seg_size,channels) 
#              containing the n training images
#   - y_train: Numpy array of size n. An element is 1 if the corresponding
#              image in x_train has an anomaly, and 0 otherwise

def training_loader(impath, maskpath, seg_size, channels):
    
    # List all files in directories
    imlist = listdir(impath)
    masklist = listdir(maskpath)

    if len(imlist) != len(masklist):
      print(f'Found {len(imlist)} images, {len(masklist)} masks. Must have equal amounts')
    else:  
      print(f'{len(imlist)} images found')
    
    # Read first image
    train_im = imread(impath+imlist[0])
    train_mask = imread(maskpath+masklist[0])
    
    # Create training images and labels for first image
    (x_train, y_train, _, _) = imgseg(train_im, train_mask, seg_size, seg_size, channels, seg_size)
    # Reshape array so images are stacked in one dimension
    x_train = x_train.reshape(-1,seg_size, seg_size,3)
    y_train = y_train.reshape(-1)
    
    # Repeat above process for each following image in the directory
    # The resulting arrays are added to the end of x_train
    if len(imlist) > 1:
      for im, mask in zip(imlist[1:], masklist[1:]):
          
          train_im = imread(impath+im)
          train_mask = imread(maskpath+mask)
          
          (x_temp, y_temp, _, _) = imgseg(train_im, train_mask, seg_size, seg_size, channels, seg_size)
          x_temp = x_train.reshape(-1,seg_size, seg_size, channels)
          y_temp = y_train.reshape(-1)
          
          x_train = np.concatenate((x_train,x_temp))
          y_train = np.concatenate((y_train,y_temp))
    
    return x_train, y_train
    
##### Function: make_generator
#
# This function creates a data generator to use in training.
# Generally one should make their own generator. This 
# function is included as a quick, easy example.
#
# See Keras documentation for preprocessing options.
    
def make_generator(model, x_train, y_train, batch_size, 
          rotation_range=0, horizontal_flip=False, width_shift_range=0, height_shift_range=0, shear_range=0):
 
    train_datagen = ImageDataGenerator(
        rotation_range=rotation_range,
        horizontal_flip=horizontal_flip,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        shear_range=shear_range
        )

    train_generator = train_datagen.flow(
        x_train,
        y_train,
        batch_size=batch_size
        )

    return train_generator

##### Function: train
#
# Similar to make_generator but for the actual training 
#
# See Keras documentation on fit for options
    
def train(model, train_generator, val_generator, epochs):
    
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator
        )

    return history