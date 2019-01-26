import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Dropout
from keras.optimizers import SGD
from keras.utils import np_utils


data = pd.read_csv('student_data.csv')
# print(data.head())

# One hot encoding
# Make dummy variables for rank
one_hot_data = pd.concat([data, pd.get_dummies(data['rank'], prefix='rank')], axis=1)

# Drop the previous rank column
one_hot_data = one_hot_data.drop('rank', axis=1)

# Print the first 10 rows of our data
# print(one_hot_data[:10])

# Scaling the data in range of 0 to 1
# Copying our data
processed_data = one_hot_data[:]

# Scaling the columns
processed_data['gre'] = processed_data['gre']/800
processed_data['gpa'] = processed_data['gpa']/4.0
# print(processed_data[:10])

# Get list of any random index
sample = np.random.choice(processed_data.index, size=int(len(processed_data)*0.9), replace=False)
# Get Training Data and Test Data
train_data, test_data = processed_data.iloc[sample], processed_data.drop(sample)

print("Number of training samples is", len(train_data))
print("Number of testing samples is", len(test_data))
print(train_data[:10])
print(test_data[:10])

# Separate data and one-hot encode the output
# Turning the data into numpy arrays, in order to train the model in Keras
features = np.array(train_data.drop('admit', axis=1)) # Take all data except admit data
targets = np.array(keras.utils.to_categorical(train_data['admit'], 2)) # Take admit data and 
features_test = np.array(test_data.drop('admit', axis=1))
targets_test = np.array(keras.utils.to_categorical(test_data['admit'], 2))

print(features[:10])
print(targets[:10])


model = Sequential()
model.add(Dense(256,activation='relu',input_shape=(6,)))
model.add(Dropout(0.3))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(2,activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(features,targets,epochs=50,batch_size=25,verbose=0)
score = model.evaluate(features,targets)
print('Training Accuracy :',score[1]*100)

score = model.evaluate(features_test, targets_test)
print('Testing Accuracy :',score[1]*100)