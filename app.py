# importing required packages and libraries

# flask for web app.
import flask as fl

# Numerical arrays
import numpy as np

# Data frames
import pandas as pd

# Scikit-Learn
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

# Neural networks
import tensorflow as tf
import tensorflow.keras as kr
from tensorflow.keras import layers

# importing the data set
url = "https://raw.githubusercontent.com/ianmcloughlin/2020A-machstat-project/master/dataset/powerproduction.csv"
df = pd.read_csv(url, error_bad_lines=True)
# removing the o values from the data set.
df = df[df != 0].dropna()
# retrieve the array.
data= df.values

# split into input and output elements.
x, y = data[:, :-1], data[:, -1]

# Splitting the data set into train and test.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=1)

# Create a neural network with neurons.
model = kr.models.Sequential()

model.add(kr.layers.Dense(200, input_shape = (1,), activation = "relu", kernel_initializer = "VarianceScaling", bias_initializer = "VarianceScaling"))
model.add(kr.layers.Dense(150, input_shape = (1,), activation = "sigmoid", kernel_initializer = "glorot_uniform", bias_initializer = "glorot_uniform"))
model.add(kr.layers.Dense(1, activation ="linear", kernel_initializer="glorot_uniform", bias_initializer="glorot_uniform"))
# Compile the model
model.compile(kr.optimizers.Adam(lr=0.002), loss = "mean_squared_error", metrics=["accuracy"], run_eagerly=True)

# Train the neural network on training data.
# We pass epochs=500 and in 70 batches.
model.fit(x_train, y_train, epochs = 500, batch_size = 70)

# creating the function

def keras(x):
    return model.predict([x])


# create a new web app.
app = fl.Flask(__name__)

# Add root route.
@app.route("/")
def home():
    return app.send_static_pages("index.html")

# Add keras route.
@app.route("/api/keras")
def keras():
    return model.predict([x])