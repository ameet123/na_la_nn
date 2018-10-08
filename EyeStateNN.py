from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from keras.layers import Dense, Dropout
from keras.models import Sequential
from numpy.random import seed
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.utils import class_weight
from sklearn.model_selection import learning_curve

seed(42)
from tensorflow import set_random_seed

set_random_seed(42)


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
    return x_train, x_test, y_train, y_test, class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)


def define_model(input_dim, in_neurons, out_neurons, hidden_dim, num_hidden_layer, is_dropout, output_act,
                 other_act='relu'):
    model = Sequential()
    model.add(Dense(in_neurons, input_dim=input_dim, kernel_initializer='normal', activation=other_act))
    if is_dropout:
        model.add(Dropout(0.2))
    for i in range(num_hidden_layer):
        model.add(Dense(hidden_dim, kernel_initializer='normal', activation=other_act))
        if is_dropout:
            model.add(Dropout(0.2))
    model.add(Dense(out_neurons, kernel_initializer='normal', activation=output_act))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Visualize loss history
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


# Global Variables
iterations = 200
batch_size = 1000
scalarX = RobustScaler()
train_file = "data\eye_state.csv"

# Execution
X_train, X_test, Y_train, Y_test, class_weights = readNp(train_file, scalarX, 14)
model = define_model(X_train.shape[1], 128, 1, 64, 10, True, 'sigmoid')
# class_weights = class_weight.compute_class_weight('balanced', np.unique(Y_train), Y_train)
start = timer()
history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=iterations, class_weight=class_weights,
                    validation_data=(X_test, Y_test), verbose=0)
print("NN Model fit in: {} sec.".format(round(timer() - start, 2)))

plot_loss_history('eye_state_test_train_loss_acc.png', history)

# Evaluation
start = timer()
score = model.evaluate(X_test, Y_test, verbose=0)
print("Model evaluation done:{} sec. Test loss:{} accuracy:{}".format(round(timer() - start, 2), score[0], score[1]))
Y_pred = np.rint(model.predict(X_test))
matrix = confusion_matrix(Y_test, Y_pred)
print(matrix)
# Visualize matrix
cm_df = pd.DataFrame(matrix, index=[i for i in "01"], columns=[i for i in "01"])
sn.set(font_scale=1.4)
sn_plot = sn.heatmap(cm_df, annot=True, annot_kws={"size": 14}, fmt='g')
plt.title("Confusion Matrix: Eye State")
plt.savefig('robust_scaling_NN_heatmap.png')
