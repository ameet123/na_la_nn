"""
HYPERPARAMETERS Tuning: 0.96
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
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn

seed(42)
set_random_seed(42)

# Global Variables
LABEL_POS = 14
iterations = 20
batch_size = 1000
train_file = "data\eye_state.csv"
scalarX = RobustScaler()
INPUT_LAYER_NEURONS = 128
NUM_HIDDEN_LAYERS = 10
HIDDEN_LAYER_UNITS = 64


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

def plot_loss_history(plot_file, history):
    training_loss = history.history['loss']
    test_loss = history.history['val_loss']
    epoch_count = range(1, len(training_loss) + 1)
    plt.title("Train vs. Test Loss/Accuracy")
    plt.plot(epoch_count, training_loss, 'r--')
    plt.plot(epoch_count, test_loss, 'b-')
    plt.plot(history.history['acc'], 'c.')
    plt.plot(history.history['val_acc'], 'm.')
    plt.legend(['Training Loss', 'Test Loss', 'train Acc', 'test Acc'], loc='center right')
    plt.xlabel('Epoch')
    plt.ylabel('Loss/Accuracy')
    plt.grid()
    plt.savefig('eye_state_test_train_loss_acc.png')
    plt.clf()
    plt.cla()
    plt.gcf().clear()
    plt.close()

def visualize_results(matrix,file):
    cm_df = pd.DataFrame(matrix, index=[i for i in "01"], columns=[i for i in "01"])
    sn.set(font_scale=1.4)
    sn_plot = sn.heatmap(cm_df, annot=True, annot_kws={"size": 14}, fmt='g')
    plt.title("Confusion Matrix: Eye State")
    plt.savefig(file)


X_train, X_test, Y_train, Y_test = readNp(train_file, scalarX, LABEL_POS)
print("Total rows:{} Train:{} Test:{} TrainingLabels:{}".format(X_train.shape[0] + X_test.shape[0], X_train.shape[0],
                                                                X_test.shape[0], Y_train.shape))

model = define_model(X_train.shape[1], INPUT_LAYER_NEURONS, 1, HIDDEN_LAYER_UNITS, NUM_HIDDEN_LAYERS, 'sigmoid')
print(model.summary())

start = timer()
history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=iterations, validation_data=(X_test, Y_test),
                    verbose=0)
print("NN Model fit in: {} sec.".format(round(timer() - start, 2)))
# Training error
plot_loss_history('eye_state_test_train_loss_acc.png', history)

# Evaluation
start = timer()
score = model.evaluate(X_test, Y_test, verbose=0)
print("Model evaluation done:{} sec. Test loss:{} accuracy:{}".format(round(timer() - start, 2), score[0], score[1]))
Y_pred = np.rint(model.predict(X_test))
matrix = confusion_matrix(Y_test, Y_pred)
print(matrix)

# Visualize matrix
visualize_results(matrix,'robust_scaling_NN_heatmap.png')