
# coding: utf-8

# In[2]:


from __future__ import print_function

import numpy as np
import keras
import gc
from itertools import izip
from sklearn.model_selection import KFold
from sklearn.cross_validation import train_test_split
import tensorflow as tf
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, Dropout
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.layers import concatenate
from keras.callbacks import EarlyStopping
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv2DTranspose
from keras.callbacks import Callback
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, f1_score, fbeta_score, precision_score, recall_score

working_path = "/home/felix/output/luna/subset0/"
final_set_path='../finalset/'


K.set_image_dim_ordering('tf')  #Using Tensorflow

img_cols = 512
img_rows = 512




smooth = 1.



def dice_coef(y_true, y_pred):
    print('Dice no np')
    print(y_true.shape)
    print(y_pred.shape)
    print('---')
        
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_np(y_true,y_pred):
    print('Dice NP')
    print(y_true.shape)
    print(y_pred.shape)
    print('---')
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def f1_score(y_true, y_pred):

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / (c2+ K.epsilon())

    # How many relevant items are selected?
    recall = c1 / (c3+ K.epsilon())

    # Calculate f1_score
    f1_score = 2 * (precision * recall) /((precision + recall)+ K.epsilon())
    return f1_score


def precision(y_true, y_pred):

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / (c2+ K.epsilon())

    return precision


def recall(y_true, y_pred):

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    recall = c1 / (c3+ K.epsilon())

    return recall



def get_model():
    inputs = Input((img_rows, img_cols,1))
    conv1 = BatchNormalization()(inputs)
    conv1 = Convolution2D(32, 3, 3,activation='relu',init='lecun_uniform', border_mode='same')(conv1)
    conv1 = BatchNormalization(axis=-1)(conv1)
    conv1 = Convolution2D(32, 3, 3,activation='relu',init='lecun_uniform', border_mode='same')(conv1) 
    conv1 = BatchNormalization(axis=-1)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3,init='lecun_uniform', border_mode='same')(pool1)
    conv2 = BatchNormalization(axis=-1)(conv2)
    conv2 = Convolution2D(64, 3, 3,activation='relu',init='lecun_uniform', border_mode='same')(conv2)
    conv2 = BatchNormalization(axis=-1)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3,activation='relu',init='lecun_uniform', border_mode='same')(pool2)
    conv3 = BatchNormalization(axis=-1)(conv3)
    conv3 = Convolution2D(128, 3, 3,activation='relu',init='lecun_uniform', border_mode='same')(conv3)
    conv3 = BatchNormalization(axis=-1)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3,activation='relu',init='lecun_uniform', border_mode='same')(pool3)
    conv4 = BatchNormalization(axis=-1)(conv4)
    conv4 = Convolution2D(256, 3, 3,activation='relu',init='lecun_uniform', border_mode='same')(conv4)
    conv4 = BatchNormalization(axis=-1)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
 

    conv5 = Convolution2D(512, 3, 3,activation='relu',init='lecun_uniform', border_mode='same')(pool4)
    conv5 = BatchNormalization(axis=-1)(conv5)
    conv5 = Convolution2D(512, 3, 3, activation='relu',init='lecun_uniform', border_mode='same')(conv5)
    conv5 = BatchNormalization(axis=-1)(conv5)
   
    
    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=3)
    conv6 = SpatialDropout2D(0.4, noise_shape=None, seed=None)(up6)
    conv6 = Convolution2D(256, 3, 3,activation='relu',init='lecun_uniform', border_mode='same')(up6)
    conv6 = Convolution2D(256, 3, 3,activation='relu',init='lecun_uniform', border_mode='same')(conv6)

    
    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=3)
    conv7 = SpatialDropout2D(0.4, noise_shape=None, seed=None)(up7)
    conv7 = Convolution2D(128, 3, 3,activation='relu',init='lecun_uniform', border_mode='same')(up7)
    conv7 = Convolution2D(128, 3, 3,activation='relu',init='lecun_uniform', border_mode='same')(conv7)

    
    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=3)
    conv8 = SpatialDropout2D(0.4, noise_shape=None, seed=None)(up8)
    conv8 = Convolution2D(64, 3, 3,activation='relu',init='lecun_uniform', border_mode='same')(up8)
    conv8 = Convolution2D(64, 3, 3,activation='relu',init='lecun_uniform', border_mode='same')(conv8)  


    
    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=3)
    conv9 = SpatialDropout2D(0.4, noise_shape=None, seed=None)(up9)
    conv9 = Convolution2D(32, 3, 3,activation='relu',init='lecun_uniform', border_mode='same')(up9)  
    conv9 = Convolution2D(32, 3, 3, activation='relu',init='lecun_uniform', border_mode='same')(conv9)


    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)
    print('Conv10: ')
    print(conv10.shape)

    model = Model(input=inputs, output=conv10)
    model.compile(optimizer=Adam(lr=1.0e-5), loss=dice_coef_loss, metrics=[dice_coef, 'accuracy', precision, recall, f1_score])
    return model

def initial_load_data():
    print('-'*30)
    
    print('Loading and preprocessing train data...')
    print('-'*30)
    working_path = "/home/felix/output/luna/subset0/"
    #Loading traning data from subset 0 to 9
    imgs_train=np.load(working_path+"trainImages.npy").astype(np.float32)
    imgs_mask_train=np.load(working_path+"trainMasks.npy").astype(np.float32)
    print(imgs_mask_train.shape)
    for x in range(1,10):
        working_path="/home/felix/output/luna/subset%d/"% (x,)
        imgs_train_temp = np.load(working_path+"trainImages.npy").astype(np.float32)
        imgs_train=np.append(imgs_train,imgs_train_temp, axis=0)
        imgs_mask_train_temp = np.load(working_path+"trainMasks.npy").astype(np.float32)
        imgs_mask_train=np.append(imgs_mask_train,imgs_mask_train_temp, axis=0)
       
     #Train and test data were initially split after preprocessing
     #Due to apparent differences between train and testa data, a final normalization 
    # step was conducted --> this is why they are being merged again
    
    for x in range(0,10):
        working_path="/home/felix/output/luna/subset%d/"% (x,)
        imgs_temp= np.load(working_path+"testImages.npy").astype(np.float32)
        imgs_train=np.append(imgs_train,imgs_temp, axis=0)
        imgs_mask_temp = np.load(working_path+"testMasks.npy").astype(np.float32)
        imgs_mask_train=np.append(imgs_mask_train,imgs_mask_temp, axis=0)
    
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean  
    imgs_train /= std
    
    #Splitting training data into training data and test data
    imgs_train, imgs_test,imgs_mask_train,imgs_mask_test_true = train_test_split(imgs_train, imgs_mask_train, test_size=0.2, random_state=42)
    
    np.save('imagesTest.npy', imgs_test)
    np.save('masksTest.npy', imgs_mask_test_true)
    np.save('imagesTrain.npy', imgs_train)
    np.save('masksTrain.npy', imgs_mask_train)
   
    
    
     
    print('Train images:')
    print(len(imgs_train))
    print('Train masks:')
    print (len(imgs_mask_train))
    print('Test images:')
    print(len(imgs_test))
    print('Test masks:')
    print (len(imgs_mask_test_true))

   
    return imgs_train, imgs_mask_train,imgs_test, imgs_mask_test_true

def load_training_data():
    
   
    imgs_train=np.load('imagesTrain.npy')
    imgs_mask_train=np.load('masksTrain.npy')
     #Splitting training data into training data and validation data
     #imgs_train, imgs_val,imgs_mask_train,imgs_mask_val = train_test_split(imgs_train, imgs_mask_train, test_size=0.2, random_state=42)
    
    print('Train images:')
    print(len(imgs_train))
    print('Train masks:')
    print (len(imgs_mask_train))
 

  
    
    return imgs_train,imgs_mask_train

    
def train(use_existing):
    
    all_imgs_train,all_imgs_mask_train=load_training_data()
    
   
    
    #using k-Fold cross validation
    kf = KFold(n_splits=5)
    #Clearing Cache
    K.clear_session()
    hist=None
    model=None
    gc.collect()
    model = get_model()
    init_weights=model.get_weights()
    
    if use_existing:
        model=load_model('./unet.h5',custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef':dice_coef, 'precision':precision, 'f1_score':f1_score, 'recall': recall, 'dice_coef_np':dice_coef_np})  
        print('model and weights loaded.')
    
    for train_index, val_index in kf.split(all_imgs_train):
        
        print("TRAIN:", train_index, "TEST:", val_index)
        imgs_train, imgs_val = all_imgs_train[train_index], all_imgs_train[val_index]
        imgs_mask_train, imgs_mask_val = all_imgs_mask_train[train_index], all_imgs_mask_train[val_index]
    
        #Augmenting training images and masks
        #  create two instances with the same arguments
        # create dictionary with the input augmentation values
        data_gen_args = dict(featurewise_center=False,
                             featurewise_std_normalization=False,
                             rotation_range=30,
                             width_shift_range=0,
                             height_shift_range=0,
                             zoom_range=0, 
                             horizontal_flip=True,
                             vertical_flip = True)
        ## use this method with both images and masks
        imgs_train_datagen = ImageDataGenerator(**data_gen_args)
        #No alterations for validation set
        #imgs_val_datagen= ImageDataGenerator(rescale=1./255)

        masks_train_datagen = ImageDataGenerator(**data_gen_args)

        #no alterations for validation set
        #imgs_mask_val_datagen=ImageDataGenerator(rescale=1./255)

        # Provide the same seed and keyword arguments to the fit and flow methods
        seed = 1
        ## fit the augmentation model to the images and masks with the same seed
        print('datagen fit...')
        imgs_train_datagen.fit(imgs_train, augment=True, seed=seed)
        masks_train_datagen.fit(imgs_mask_train, augment=True, seed=seed)
        #imgs_val_datagen.fit(imgs_val, augment=False)
        #imgs_mask_val_datagen.fit(imgs_mask_val, augment=False)
        print('datagen fit done.')
        ## set the parameters for the data to come from (images)
        batch_size=2
        imgs_train_generator = imgs_train_datagen.flow(
            imgs_train,
            batch_size=batch_size,
            shuffle=True,
            seed=seed)
        ## set the parameters for the data to come from (masks)
        masks_train_generator= masks_train_datagen.flow(
            imgs_mask_train,
            batch_size=batch_size,
            shuffle=True,
            seed=seed)

        #imgs_val_generator = imgs_val_datagen.flow(
        #    imgs_val,
        #    batch_size=batch_size,
        #    shuffle=True,
        #    seed=seed)
        ## set the parameters for the data to come from (masks)
        # masks_val_generator =  imgs_mask_val_datagen.flow(
        #    imgs_mask_val,
        #   batch_size=batch_size,
        #  shuffle=True,
        # seed=seed)

        # combine generators into one which yields image and masks
        #train_generator = zip(imgs_train_generator, masks_train_generator)
        train_generator=izip(imgs_train_generator, masks_train_generator)
        #val_generator = zip(imgs_val_generator, masks_val_generator)
        #val_generator=izip(imgs_val_generator, masks_val_generator)

        print('-'*30)
        print('Creating and compiling model...')
        print('-'*30)
        #Clearing Session infos, to clear GPU from stale models
        



        # Saving weights to unet.hdf5 at checkpoints
        model_checkpoint = ModelCheckpoint('unet.h5', monitor='val_loss', save_best_only=True)
        #
        # Should we load existing weights? 
        # Set argument for call to train_and_predict to true at end of script
      


        print('Fitting model...')
        #Early Stop when Validation does not decrease anymore
        #early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    
        #Reduce learning rate by a certain factor when learning stagnates
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=6, min_lr=0.001)

        tbCallBack= keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=2, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

        steps_per_epoch=len(imgs_train)
        #validation_steps=len(imgs_val)/batch_size
        hist=model.fit_generator(train_generator,
                        steps_per_epoch=steps_per_epoch,
                        validation_data=[imgs_val, imgs_mask_val],
                        epochs=10,verbose=1,callbacks=[model_checkpoint, tbCallBack, reduce_lr])

       
       
      
        print(hist.history)
       
    #Resetting weights, so no stale models would linger in GPU memory
    model.set_weights(init_weights)
    
def predict():
    # loading best weights from training session
  
    model=load_model('./unet.h5',custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef':dice_coef, 'precision':precision, 'f1_score':f1_score, 'recall': recall, 'dice_coef_np':dice_coef_np})  
    print('model and weights loaded.')
    
    print('Predicting masks on test data...')
    imgs_test=np.load('imagesTest.npy')
    imgs_mask_test_true=np.load('masksTest.npy')
    num_test = len(imgs_test)
    print('Test images: %d',num_test)
    imgs_mask_test = np.ndarray([num_test,512,512,1],dtype=np.float32)
    for i in range(num_test):
        imgs_mask_test[i] = model.predict([imgs_test[i:i+1]], verbose=0)[0]
    print('PredMask: ')
    print(imgs_mask_test.shape)
    np.save('masksTestPredicted.npy', imgs_mask_test)
   
    
    mean = 0.0

    for i in range(num_test):
        
       # Channel last
       # mean+=dice_coef_np(imgs_mask_test_true[i,0], imgs_mask_test[i,0])
        mean+=dice_coef_np(imgs_mask_test_true[i], imgs_mask_test[i]) 
            
        
    mean/=num_test
    print("Mean Dice Coeff : ",mean)

if __name__ == '__main__':
    for x in range(1):
        #initial_load_data()
        train(False)
        #predict()
        

