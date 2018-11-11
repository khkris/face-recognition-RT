

from ModelKK import *

import numpy as np
import pickle

from keras.models import Sequential
from keras.layers import *
from keras.models import Model
from keras.preprocessing.image import *
from keras.applications.imagenet_utils import preprocess_input
import tensorflow as tf

#====================

# Call in Database for reading
pickle_in = open("DB.pickle", "rb")
Database = pickle.load(pickle_in)

def img_to_encoding(img, model):
    
    # Returns the encoding of the input image.
    encoder = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
    encoding = encoder.predict(img)[0,:]
    
    return encoding


def img_preprocess(img_path):
    
    # Pre-processes the image to be of size 224 x 224 x 3
    # and to be fed to the model.
    img = load_img(img_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


def recognition(img, Database, model):

    # Encode the input image
    encoding = img_to_encoding(img, model)

    # Set the minimum distance to a high value
    min_dist = 1000

    #Calculate the L2 norm between encoding of the input and the database encodings
    for (name, DB_encoding) in Database.items():
         
        dist = np.linalg.norm( DB_encoding - encoding )

        if dist < min_dist:
            min_dist = dist
            identity = name
            
        
    if min_dist > 68.5:
            identity = "Unknown"
    
    return min_dist, identity

# Add encoding of an image to the database
def add_to_database(name, img, model):
    
    Database[str(name)] = img_to_encoding(img, model)

