
# coding: utf-8

# In[ ]:


from __future__ import print_function

import numpy as np
import keras
from itertools import izip
import tensorflow as tf
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, Dropout
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import Callback
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, f1_score, fbeta_score, precision_score, recall_score

working_path = "/home/felix/output/luna/subset0/"



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
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_last')(inputs)
    conv1= BatchNormalization()
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv1)
    conv1= BatchNormalization()
    pool1 = MaxPooling2D((2, 2))(conv1)
    #
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_last')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    #
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_last')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv3)

    up1 = UpSampling2D(size=(2, 2))(conv3)
    up1 = concatenate([conv2,up1],axis=3)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_last')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv4)
    #
    up2 = UpSampling2D(size=(2, 2))(conv4)
    up2 = concatenate([conv1,up2], axis=3)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_last')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv5)
    
      
    conv6 = Convolution2D(1, 1, 1, activation='sigmoid')(conv5)

    model = Model(input=inputs, output=conv6)


    model.compile(optimizer=Adam(lr=1.0e-4), loss=dice_coef_loss, metrics=[dice_coef, 'accuracy', precision, recall, f1_score])
    return model

def load_data():
    print('-'*30)
    
    print('Loading and preprocessing train data...')
    print('-'*30)
    working_path = "/home/felix/output/luna/subset1/"
    #Loading traning data from subset 1 to 9
    imgs_train=np.load(working_path+"trainImages.npy").astype(np.float32)
    imgs_mask_train=np.load(working_path+"trainMasks.npy").astype(np.float32)
    print(imgs_mask_train.shape)
    for x in range(2,10):
        working_path="/home/felix/output/luna/subset%d/"% (x,)
        imgs_train_temp = np.load(working_path+"trainImages.npy").astype(np.float32)
        imgs_train=np.append(imgs_train,imgs_train_temp, axis=0)

        imgs_mask_train_temp = np.load(working_path+"trainMasks.npy").astype(np.float32)
        imgs_mask_train=np.append(imgs_mask_train,imgs_mask_train_temp, axis=0)
       
     
    #Using the training data from subset 0 for validation
    imgs_val =np.load("/home/felix/output/luna/subset0/trainImages.npy").astype(np.float32)
    imgs_mask_val=np.load("/home/felix/output/luna/subset0/trainMasks.npy").astype(np.float32)
    #imgs_val_temp = np.load("/home/felix/output/luna/subset9/trainImages.npy").astype(np.float32)
    #imgs_val=np.append(imgs_val,imgs_val_temp, axis=0)
    #imgs_mask_val_temp = np.load("/home/felix/output/luna/subset9/trainMasks.npy").astype(np.float32)
    #imgs_mask_val=np.append(imgs_mask_val,imgs_mask_val_temp, axis=0)
    
    #Loading test data from subset 0-9
    working_path = "/home/felix/output/luna/subset0/"
    imgs_test =np.load(working_path+"testImages.npy").astype(np.float32)
    imgs_mask_test_true =np.load(working_path+"testMasks.npy").astype(np.float32)
     
    for x in range(1,10):
        working_path="/home/felix/output/luna/subset%d/"% (x,)
        imgs_temp= np.load(working_path+"testImages.npy").astype(np.float32)
        imgs_test=np.append(imgs_test,imgs_temp, axis=0)
        imgs_mask_temp = np.load(working_path+"testMasks.npy").astype(np.float32)
        imgs_mask_test_true=np.append(imgs_mask_test_true,imgs_mask_temp, axis=0)
    
    
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean  
    imgs_train /= std

    return imgs_train, imgs_mask_train, imgs_val, imgs_mask_val, imgs_test, imgs_mask_test_true

    
def train(use_existing):
    
    imgs_train, imgs_mask_train, imgs_val, imgs_mask_val, imgs_test, imgs_mask_test_true=load_data()
    print('daten:')
    print(len(imgs_train))
    print (len(imgs_mask_train))
    print(len(imgs_val))
    print(len(imgs_mask_val))
    print(len(imgs_test))
    print (len(imgs_mask_test_true))
    
    
    #Augmenting training images and masks
    #  create two instances with the same arguments
    # create dictionary with the input augmentation values
    data_gen_args = dict(featurewise_center=True,
                         featurewise_std_normalization=True,
                         rotation_range=0,
                         width_shift_range=0,
                         height_shift_range=0,
                         zoom_range=0.2, 
                         horizontal_flip=False,
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
    K.clear_session()
    model = None
    model = get_model()
   
    # Saving weights to unet.hdf5 at checkpoints
    model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss', save_best_only=True)
    #
    # Should we load existing weights? 
    # Set argument for call to train_and_predict to true at end of script
    if use_existing:
        model.load_weights('./unet.hdf5')  
   
   
    print('Fitting model...')
    #Early Stop when Validation does not decrease anymore
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    
    tbCallBack= keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=2, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    
    steps_per_epoch=len(imgs_train)/batch_size
    #validation_steps=len(imgs_val)/batch_size
    hist=model.fit_generator(train_generator,
                    steps_per_epoch=steps_per_epoch,
                    validation_data=[imgs_val, imgs_mask_val],
                    epochs=25,verbose=1,callbacks=[model_checkpoint, tbCallBack, early_stopping])
    
    '''
    tbCallBack= keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=2, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    hist=model.fit_generator(train_generator, steps_per_epoch=2000, epochs=50)
    model.fit(imgs_train, imgs_mask_train,validation_split=0.1, batch_size=2, epochs=10, verbose=1, shuffle=True,
              callbacks=[model_checkpoint, tbCallBack])
    '''
    print(hist.history)
  

    
def predict(imgs_test, imgs_mask_test_true, model):
    # loading best weights from training session
    print('Loading saved weights...')
    model.load_weights('./unet.hdf5')

    print('Predicting masks on test data...')
    num_test = len(imgs_test)
    imgs_mask_test = np.ndarray([num_test,512,512,1],dtype=np.float32)
    for i in range(num_test):
        imgs_mask_test[i] = model.predict([imgs_test[i:i+1]], verbose=0)[0]
    print('PredMask: ')
    print(imgs_mask_test.shape)
    np.save('masksTestPredicted.npy', imgs_mask_test)
    np.save('imagesTest.npy', imgs_test)
    np.save('masksTestTrue.npy', imgs_mask_test_true)
    
    mean = 0.0
    print('Mask_test_true: ')
    print(imgs_mask_test_true[i][:, :, :].shape)
    print('Mask_test_true: ')
    print(imgs_mask_test_true[i].shape)
    print('Mask_test_true: ')
    print(imgs_mask_test_true[i,0].shape)
    for i in range(num_test):
        
       # Channel last
       # mean+=dice_coef_np(imgs_mask_test_true[i,0], imgs_mask_test[i,0])
        mean+=dice_coef_np(imgs_mask_test_true[i], imgs_mask_test[i]) 
            
    #mean=dice_coef(imgs_mask_test_true, imgs_mask_test)        
    mean/=num_test
    print("Mean Dice Coeff : ",mean)

if __name__ == '__main__':
    for x in range(1):
        train(False)
        #imgs_train, imgs_mask_train, imgs_test, imgs_mask_test_true=load_data()
        #model=None
        #model=get_model()
        #predict(imgs_test, imgs_mask_test_true, model)
        #load_data()
        

