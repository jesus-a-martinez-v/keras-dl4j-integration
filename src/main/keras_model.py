import numpy as np
import pandas
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

# Fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# Load dataset
dataframe = pandas.read_csv("./resources/iris.data", header=None)
dataframe = shuffle(dataframe)
dataset = dataframe.values

features = dataset[:, 0:4].astype(float)
labels = dataset[:, 4]

print(features)
print(labels)

# Encode class values as integers
encoder = LabelEncoder()
encoder.fit(labels)
print(encoder.get_params())
encoded_labels = encoder.transform(labels)

# Convert integers to dummy variables (one hot encoding)
one_hot_labels = np_utils.to_categorical(encoded_labels)
print(one_hot_labels)

# Actual model
model = Sequential()
model.add(Dense(4, input_dim=4, activation="relu"))
model.add(Dense(3, activation="sigmoid"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Fit the model.
model.fit(features, one_hot_labels, nb_epoch=200, batch_size=5)


# Save the model
model.save_weights("iris_model_save")
model_as_json = model.to_json()

with open("iris_model_json", "w") as f:
	f.write(model_as_json)

	
