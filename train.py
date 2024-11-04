import os
import numpy as np
from tensorflow.keras.layers import Bidirectional, Dense, Flatten, LSTM, Input, Conv1D, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.optimizers import Adam
from data_preparation import prepare_data
from utils import my_loss, predict_point_by_point, MAE, MAPE, RMSE
from attention import AttentionLayer
from attention_with_context import AttentionWithContext
from shuffle_attention import ShuffleAttention

# Set environment to control device usage if needed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Path to your data file
file_path = '/content/drive/MyDrive/CVIP/Revised Code /Updated_Combined_Traffic_Data_1.xlsx'

# Load and prepare the data from the Excel file
train_data, train_target, test_data, test_target, encoder, scaler = prepare_data(file_path)

# Reshape data as required by model layers
train_data = np.reshape(train_data, (train_data.shape[0], train_data.shape[1], train_data.shape[2], 1))
test_data = np.reshape(test_data, (test_data.shape[0], test_data.shape[1], test_data.shape[2], 1))

# Model parameters
epochs = 50
batch_size = 128
hidden_dim = 15

# Model architecture with Conv-LSTM, Shuffle Attention, and dual Bi-LSTM networks

# Conv-LSTM Input and Conv layers
main_input = Input(shape=(train_data.shape[1], train_data.shape[2], 1), name='main_input')
con1 = TimeDistributed(Conv1D(filters=hidden_dim, kernel_size=3, padding='same', activation='relu', strides=1))(main_input)
con2 = TimeDistributed(Conv1D(filters=hidden_dim, kernel_size=3, padding='same', activation='relu', strides=1))(con1)
sa_layer = TimeDistributed(ShuffleAttention(groups=2, channels=hidden_dim))(con2)
con_fl = TimeDistributed(Flatten())(sa_layer)
con_out = Dense(hidden_dim)(con_fl)

# Weekday Bi-LSTM with Attention
auxiliary_input_w = Input((15, 1), name='auxiliary_input_w')
lstm_outw1 = Bidirectional(LSTM(hidden_dim, return_sequences=True))(auxiliary_input_w)
weekday_attention = AttentionWithContext(name="weekday_attention")(lstm_outw1)
lstm_outw2 = Bidirectional(LSTM(hidden_dim, return_sequences=False))(weekday_attention)

# Weekend Bi-LSTM with Attention
auxiliary_input_d = Input((15, 1), name='auxiliary_input_d')
lstm_outd1 = Bidirectional(LSTM(hidden_dim, return_sequences=True))(auxiliary_input_d)
weekend_attention = AttentionWithContext(name="weekend_attention")(lstm_outd1)
lstm_outd2 = Bidirectional(LSTM(hidden_dim, return_sequences=False))(weekend_attention)

# Concatenate and Final Dense Layers
combined_output = Dense(20, activation='relu')(keras.layers.concatenate([lstm_outw2, lstm_outd2]))
main_output = Dense(1, activation='relu', kernel_regularizer=l1_l2(0.1, 0.1), name='main_output')(combined_output)

# Define the model with inputs and outputs
model = Model(inputs=[main_input, auxiliary_input_w, auxiliary_input_d], outputs=main_output)

# Compile with Adam optimizer and custom loss
model.compile(optimizer=Adam(), loss=my_loss)

# Define checkpoint callback
checkpoint = ModelCheckpoint("model/model_{epoch:02d}-{val_loss:.4f}.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# Train the model
model.fit([train_data, train_data, train_data], train_target,
          batch_size=batch_size, epochs=epochs, validation_split=0.15, verbose=2,
          callbacks=callbacks_list)

# Predict and Evaluate
predicted = predict_point_by_point(model, [test_data, test_data, test_data])
p_real, l_real = scaler.inverse_transform(predicted), scaler.inverse_transform(test_target)
print("MAE:", MAE(p_real, l_real))
print("MAPE:", MAPE(p_real, l_real))
print("RMSE:", RMSE(p_real, l_real))
