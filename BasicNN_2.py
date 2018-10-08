"""
SCALING: 0.86
"""

from timeit import default_timer as timer

import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from numpy.random import seed
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from tensorflow import set_random_seed

seed(42)
set_random_seed(42)

# Global Variables
LABEL_POS = 14
iterations = 20
batch_size = 1000
train_file = "data\eye_state.csv"
scalarX = RobustScaler()
INPUT_LAYER_NEURONS = 32
NUM_HIDDEN_LAYERS = 2
HIDDEN_LAYER_UNITS = 16


# Functions
def readNp(file, scalarX, label_pos):
    data = np.loadtxt(file, delimiter=',', skiprows=1)
    X = data[:, 0:label_pos]
    Y = data[:, label_pos]
    scalarX = scalarX.fit(X)
    X = scalarX.transform(X)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=7)
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    return x_train, x_test, y_train, y_test


def define_model(input_dim, in_neurons, out_neurons, hidden_dim, num_hidden_layer, output_act):
    model = Sequential()
    model.add(Dense(in_neurons, input_dim=input_dim, kernel_initializer='normal', activation='relu'))
    for i in range(num_hidden_layer):
        model.add(Dense(hidden_dim, activation='relu'))
    model.add(Dense(out_neurons, activation=output_act))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


X_train, X_test, Y_train, Y_test = readNp(train_file, scalarX, LABEL_POS)
print("Total rows:{} Train:{} Test:{} TrainingLabels:{}".format(X_train.shape[0] + X_test.shape[0], X_train.shape[0],
                                                                X_test.shape[0], Y_train.shape))

model = define_model(X_train.shape[1], INPUT_LAYER_NEURONS, 1, HIDDEN_LAYER_UNITS, NUM_HIDDEN_LAYERS, 'sigmoid')
print(model.summary())

start = timer()
history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=iterations, validation_data=(X_test, Y_test),
                    verbose=0)
print("NN Model fit in: {} sec.".format(round(timer() - start, 2)))
# Evaluation
start = timer()
score = model.evaluate(X_test, Y_test, verbose=0)
print("Model evaluation done:{} sec. Test loss:{} accuracy:{}".format(round(timer() - start, 2), score[0], score[1]))
Y_pred = np.rint(model.predict(X_test))
matrix = confusion_matrix(Y_test, Y_pred)
print(matrix)
