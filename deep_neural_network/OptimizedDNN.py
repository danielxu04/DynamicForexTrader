
import random
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.regularizers import l1, l2
from keras.optimizers import Adam
from scikeras.wrappers import KerasClassifier

def set_seeds(seed = 100):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
def cw(df):
    c0, c1 = np.bincount(df['Direction'])
    w0 = (1/c0) * (len(df)) / 2
    w1 = (1/c1) * (len(df)) / 2
    return {0:w0, 1:w1}

# 2 layers, 100 nodes by default
def create_model(hl=3, hu=50, dropout=False, rate=0.3, regularize=False,
                 reg=l1(0.0005), input_dim=None):
    if not regularize:
        reg = None
    
    model = Sequential()
    model.add(Dense(hu, input_dim=input_dim, activity_regularizer=reg, activation="relu"))
    
    if dropout:
        model.add(Dropout(rate, seed=100))
    
    for layer in range(hl):
        model.add(Dense(hu, activation="relu", activity_regularizer=reg))
        if use_dropout:
            model.add(Dropout(rate, seed=100))
    
    model.add(Dense(1, activation="sigmoid"))
    optimizer = Adam(learning_rate = 0.0001)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model


# Create KerasClassifier
def create_keras_model(hl=3, hu=50, dropout=False, rate=0.3, regularize=False,
                       reg=l1(0.0005), input_dim=None):
    return KerasClassifier(build_fn=create_model,
                           hl=hl, hu=hu, dropout=dropout, rate=rate, regularize=regularize,
                           reg=reg, input_dim=input_dim, epochs=150, batch_size=32, verbose=0)


