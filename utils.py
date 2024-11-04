import numpy as np
from tensorflow.keras import backend as K

def rmse_train(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def my_loss(y_true, y_pred):
    mse = K.mean(K.square(y_true - y_pred), axis=-1)
    return mse

def predict_point_by_point(model, data):
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    print("predict_size:", predicted.size)
    return predicted

def MAE(predicted, true):
    return np.mean(np.abs(predicted - true))

def MAPE(predicted, true):
    return np.mean(np.abs((predicted - true) / true)) * 100

def RMSE(predicted, true):
    return np.sqrt(np.mean(np.square(predicted - true)))
