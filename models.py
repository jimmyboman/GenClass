import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Flatten, Reshape, Dense, Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from tensorflow.keras.models import Model

##### Function: binary_cnn
# Takes input image dimensions and builds a binary classifier network
# for transfer learning. The model is based on the VGG19 network without
# the top layer, and instead has a global average pooling layer and a 
# dense layer for binary classification. Returns the compiled model.
def binary_cnn(img_h,img_w,channels,learning_rate):
    
    IMG_SHAPE = (img_h,img_w,channels)
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    base_model = tf.keras.applications.vgg19.VGG19(input_shape=IMG_SHAPE,
                                                            include_top=False,
                                                            weights='imagenet')
                                                            
    base_model.trainable = False
    
    model = tf.keras.Sequential([
                                base_model,
                                tf.keras.layers.GlobalAveragePooling2D(),
                                tf.keras.layers.Dense(1)])
                                
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy'])
                  
    return model
    
##### Function: autoencoder
# Takes input image dimensions and builds an autoencoder network.
# Returns the compiled model.                  
def autoencoder(img_h,img_w,channels):

    optimizer = 'Adam'
    loss = 'mse'
    
    input_img = Input(shape=(img_h, img_w, channels))  # adapt this if using `channels_first` image data format
    
    # encode
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    # at this point the representation is (4, 4, 8) i.e. 128-dimensional
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # flatten and compress
    shape = K.int_shape(encoded) 
    bn = Flatten()(encoded)
    bn = Dense(2)(bn)
    bnRec = Dense(shape[1] * shape[2] * shape[3], activation='relu')(bn)
    encoded = Reshape((shape[1], shape[2], shape[3]))(bnRec)

    # decode
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = ZeroPadding2D((1,1))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    # create encoder
    encoder = Model(input_img, bn)
    # create autoencoder
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer=optimizer, loss=loss)
    
    return autoencoder
	
    