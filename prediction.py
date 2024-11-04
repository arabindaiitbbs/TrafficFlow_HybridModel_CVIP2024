import numpy as np
import pandas as pd
from tensorflow.keras.models import model_from_json
from data_preparation import prepare_data
from utils import predict_point_by_point, MAE, MAPE, RMSE

# Path to your data and model files
file_path = '/content/drive/MyDrive/CVIP/Revised Code/Updated_Combined_Traffic_Data_1.xlsx'
model_json_path = 'model/conv_lstm.json'
model_weights_path = 'model/model_best.h5'  # replace 'model_best.h5' with the correct file name 

# Load and prepare the test data
_, _, test_data, test_target, encoder, scaler = prepare_data(file_path)

# Reshape test data as required by model layers
test_data = np.reshape(test_data, (test_data.shape[0], test_data.shape[1], test_data.shape[2], 1))

# Load model architecture and weights
with open(model_json_path, 'r') as json_file:
    loaded_model_json = json_file.read()

# Load model from JSON
model = model_from_json(loaded_model_json, custom_objects={"AttentionLayer": AttentionLayer, 
                                                           "AttentionWithContext": AttentionWithContext, 
                                                           "ShuffleAttention": ShuffleAttention})

# Load weights into the model
model.load_weights(model_weights_path)
print("Model loaded from disk.")

# Compile the model (required for making predictions)
model.compile(optimizer='adam', loss='mse')

# Make predictions on test data
predicted = predict_point_by_point(model, [test_data, test_data, test_data])  # Adjust input structure if necessary

# Rescale predictions and test labels back to original scale
p_real = scaler.inverse_transform(predicted.reshape(-1, 1)).flatten()
l_real = scaler.inverse_transform(test_target.reshape(-1, 1)).flatten()

# Evaluate the model
mae = MAE(p_real, l_real)
mape = MAPE(p_real, l_real)
rmse = RMSE(p_real, l_real)

# Print evaluation metrics
print("MAE:", mae)
print("MAPE:", mape)
print("RMSE:", rmse)
