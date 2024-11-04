import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from tensorflow.keras.utils import to_categorical

# Define the sequence length, e.g., 15 intervals for 75 minutes of historical data
SEQ_LEN = 15

def preprocess_timestamps(data):
    """
    Preprocesses the timestamp data to add cyclical time-based features.
    
    Args:
        data (DataFrame): The traffic data with a 'timeperiod' column.
        
    Returns:
        DataFrame: Data with additional cyclical time-based features.
    """
    # Extract time components
    data['hour'] = data['timeperiod'].dt.hour
    data['day_of_week'] = data['timeperiod'].dt.dayofweek
    data['month'] = data['timeperiod'].dt.month
    
    # Create cyclical features for hours, day of the week, and month
    data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
    data['day_of_week_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
    data['day_of_week_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
    data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
    data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
    
    # Drop original columns to avoid redundancy
    data.drop(['timeperiod', 'hour', 'day_of_week', 'month'], axis=1, inplace=True)
    return data

def load_and_preprocess_data(file_path):
    """
    Loads and preprocesses the traffic data from an Excel file.
    
    Args:
        file_path (str): Path to the Excel file containing traffic data.
        
    Returns:
        Tuple: Preprocessed data, target labels, encoder, and scaler.
    """
    # Load data and parse the 'timeperiod' column as datetime
    data = pd.read_excel(file_path)
    data['timeperiod'] = pd.to_datetime(data['timeperiod'])
    
    # Drop columns that may not be useful for model training
    if 'speed_limit' in data.columns:
        data.drop('speed_limit', axis=1, inplace=True)
    
    # Preprocess time-based features
    data = preprocess_timestamps(data)
    
    # Separate the target variable if it exists, otherwise set it as None
    target = data.pop('los').astype('category').cat.codes if 'los' in data.columns else None
    
    # Identify categorical and numerical features
    categorical_features = ['location', 'location_name', 'number_of_day', 'day', 'weather', 'holiday']
    numerical_features = [col for col in data.columns if col not in categorical_features]
    
    # Scale numerical features and encode categorical features
    scaler = MinMaxScaler()
    data[numerical_features] = scaler.fit_transform(data[numerical_features])
    
    encoder = OneHotEncoder(sparse_output=False)
    encoded_data = encoder.fit_transform(data[categorical_features])
    encoded_features = encoder.get_feature_names_out(categorical_features)
    
    # Concatenate the encoded features with numerical features
    data = data.drop(categorical_features, axis=1)
    data[encoded_features] = encoded_data
    
    return data, target, encoder, scaler

def create_sequences(data, target, seq_len=SEQ_LEN):
    """
    Creates sequences of data for time series modeling.
    
    Args:
        data (DataFrame): The preprocessed data.
        target (array): Target values.
        seq_len (int): The sequence length for each sample.
        
    Returns:
        Tuple: Arrays of data sequences and corresponding labels.
    """
    sequences, labels = [], []
    
    for i in range(len(data) - seq_len):
        sequences.append(data.iloc[i:i + seq_len].values)
        labels.append(target[i + seq_len])
    
    return np.array(sequences), np.array(labels)

def stratified_temporal_split(data, target, seq_len=SEQ_LEN, test_size=0.2):
    """
    Splits the data into training and testing sets in a temporal order.
    
    Args:
        data (DataFrame): The preprocessed data.
        target (array): Target values.
        seq_len (int): The sequence length for each sample.
        test_size (float): The proportion of data to use for testing.
        
    Returns:
        Tuple: Train and test data and target sets.
    """
    split_index = int((1 - test_size) * (len(data) - seq_len))
    train_data, train_target = create_sequences(data.iloc[:split_index + seq_len], target[:split_index + seq_len], seq_len)
    test_data, test_target = create_sequences(data.iloc[split_index:], target[split_index:], seq_len)
    
    return train_data, train_target, test_data, test_target

def prepare_data(file_path):
    """
    Prepares the data for model training by loading, preprocessing, and splitting.
    
    Args:
        file_path (str): Path to the Excel file containing traffic data.
        
    Returns:
        Tuple: Train data, train target, test data, test target, encoder, and scaler.
    """
    # Load and preprocess the data
    data, target, encoder, scaler = load_and_preprocess_data(file_path)
    
    # Convert target to categorical format for training
    target_categorical = to_categorical(target) if target is not None else None
    
    # Split into training and testing sets
    train_data, train_target, test_data, test_target = stratified_temporal_split(data, target_categorical, SEQ_LEN)
    
    return train_data, train_target, test_data, test_target, encoder, scaler

# Main entry point for testing or standalone execution
if __name__ == "__main__":
    file_path = '/content/drive/MyDrive/CVIP/Revised Code /Updated_Combined_Traffic_Data_1.xlsx'
    train_data, train_target, test_data, test_target, encoder, scaler = prepare_data(file_path)
    print("Data preparation completed successfully.")
