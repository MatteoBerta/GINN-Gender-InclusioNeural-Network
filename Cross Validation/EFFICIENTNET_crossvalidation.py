import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd 
import os
import shutil
import cv2
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import KFold, StratifiedKFold

from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint
import keras.utils
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from keras.utils import to_categorical

import efficientnet.keras as efn

data_folder = " " #insert your df folder here
image_dir = " " #insert your image folder here

###############################
###### HYPERPARAMETERS ########
###############################

batch_size = 64
target_size = (224, 224)
input_shape=(224, 224, 3)
seed=1337
adam = 0.001
fre= -20
FC = 2048
patience = 3
verbose = 1
factor = 0.50
min_lr = 0.0001
EPOCHS = 5 
N_SPLITS = 5 

def read_dataframe(path_to_data):
    df = pd.read_csv(path_to_data + '\list_partition.csv')
    df = df.drop('Unnamed: 0', axis=1) 
    df.loc[:, 'gender'] = df['gender'].astype(str)
    return df 

def K_fold(dataframe_): 
    main_prediction = []
    data_kfold = pd.DataFrame()

    #Initializaing data generators
    #Data Augmentation for the train set
    train_datagen = ImageDataGenerator(rescale= 1./255,
                                       shear_range=0.2, 
                                       zoom_range=0.2, 
                                       horizontal_flip=True)
    #Rescaling for validation set
    validation_datagen = ImageDataGenerator(rescale=1./255)

    #Inizialization of kfold 
    kf = StratifiedKFold(n_splits = N_SPLITS, shuffle = True, random_state = 42)
    Y = dataframe_[['gender']]
    #Convert integer labels to string labels 
    Y.loc[:, 'gender'] = Y['gender'].astype(str)

    return kf, Y, train_datagen, validation_datagen 


def get_model_name(k): 
    return 'model_'+str(k)+'.h5'

def train_model(df, train_datagen, validation_datagen): 
    VALIDATION_ACCURACY = []
    VALIDATION_LOSS = []

    train_y = df['gender']
    train_x = df['image']
    save_dir = '/saved_models/'
    fold_var = 1

    for train_index, val_index in kf.split(np.zeros(train_x.shape[0]),Y):

        print("--------------------")
        print("Fold:", fold_var) 
        
        training_data = df.iloc[train_index]
        validation_data = df.iloc[val_index]
        
        train_data_generator = train_datagen.flow_from_dataframe(training_data, directory = image_dir,
                                x_col = "image", y_col = "gender",
                                class_mode = "categorical", shuffle = True, target_size = (224, 224))
        valid_data_generator  = validation_datagen.flow_from_dataframe(validation_data, directory = image_dir,
                                x_col = "image", y_col = "gender",
                                class_mode = "categorical", shuffle = True, target_size = (224, 224))
        
        # CREATE NEW MODEL
        base_model = efn.EfficientNetB4(input_shape = (224, 224, 3), include_top = False, weights = 'imagenet') ## IN THIS CASE IS EFFICIENTNET 
        
        # Freezing Layers
        for layer in base_model.layers:
            layer.trainable = False   
        
        model = tf.keras.Sequential()
        model.add(base_model)
        model.add(Flatten())
        model.add(Dense(1024, activation="relu"))
        model.add(Dense(3, activation="softmax"))
    
        #Compile the model 
        model.compile(optimizer= "adam", loss="categorical_crossentropy",metrics=['accuracy'])

        #CREATE CALLBACKS
        checkpoint = tf.keras.callbacks.ModelCheckpoint(save_dir+get_model_name(fold_var), 
                                monitor='val_accuracy', verbose=1, 
                                save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        
        # Model Checkpoint
        model_path='./output/gender_model.h5'
        checkpointer = ModelCheckpoint(model_path, monitor='loss',verbose=1,save_best_only=True,
                                    save_weights_only=False, mode='auto',save_freq='epoch')
        callback_list=[checkpointer]
        
        # There can be other callbacks, but just showing one because it involves the model name
        # This saves the best model
        # FIT THE MODEL
        history = model.fit(train_data_generator,
                    epochs=EPOCHS,
                    callbacks=callback_list,
                    validation_data=valid_data_generator)
        #PLOT HISTORY
        #		:
        #		:
        
        # LOAD BEST MODEL to evaluate the performance of the model
        model.load_weights("./output/gender_model.h5")
        
        results = model.evaluate(valid_data_generator)
        results = dict(zip(model.metrics_names,results))
        
        VALIDATION_ACCURACY.append(results['accuracy'])
        VALIDATION_LOSS.append(results['loss'])
        
        tf.keras.backend.clear_session()
        
        fold_var += 1


###############################
########### MAIN ##############
###############################

df = read_dataframe(data_folder)
kf, Y, train_datagen, validation_datagen = K_fold(df)
train_model(df, train_datagen=train_datagen, validation_datagen=validation_datagen)