
# Traffic Flow Prediction with Attention-Based Conv-LSTM and Bi-LSTM Model

This repository contains the code for a hybrid deep learning model designed to predict traffic flow using a combination of convolutional and LSTM layers, with attention mechanisms to improve focus on significant features. The model leverages Conv-LSTM for spatio-temporal feature extraction, Shuffle Attention for enhanced feature discrimination, and dual Bi-LSTM networks with contextual attention to capture weekday and weekend patterns.

## Table of Contents

1. Overview
2. Features
3. Project Structure
4. Prerequisites
5. Setup
6. Usage
7. Evaluation
8. Results
9. Acknowledgments
## Overview
Traffic congestion forecasting is crucial for intelligent transportation systems. This project aims to provide accurate short-term predictions of traffic flow by using a specialized neural network architecture. 
The model utilizes:

1. #### Conv-LSTM Module: For capturing spatio-temporal patterns.
2. #### Shuffle Attention: To dynamically adjust focus on informative channels and spatial areas.
3. #### Bi-LSTM Networks: To separately capture weekday and weekend traffic patterns.
4. #### Attention Mechanism: Enhances the model’s focus on critical time segments within each Bi-LSTM module.
## Features

1. Spatio-temporal modeling with Conv-LSTM
2. Channel and spatial attention through Shuffle Attention
3. Dual Bi-LSTM modules to separately model weekday and weekend traffic patterns
4. Supports sequence length customization for historical data intervals
Modular code structure for easy customization and extension
## Prerequisites

1. Python 3.8 or higher
2. TensorFlow 2.x
3. Other Python packages as specified in requirements.txt
## Setup

1. #### git clone https://github.com/arabindaiitbbs/TrafficFlow_HybridModel_CVIP2024.git
2. #### cd TrafficFlow_HybridModel_CVIP2024

### Data Preparation:
1. Place your traffic data in Excel format (e.g., Updated_Combined_Traffic_Data_1.xlsx) in the designated directory.
2. The data file should contain columns like timeperiod, location, weather, and holiday.

### Preprocess and Split the Data:
1. Run the data preparation script within train.py, which will load and preprocess your data, scaling and encoding it for model training.

## Training

### python train.py

## Prediction
After training, use prediction.py to evaluate the model on test data:

### python prediction.py

## Evaluation
The model’s performance is evaluated using the following metrics:
1. MAE (Mean Absolute Error)
2. RMSE (Root Mean Square Error)

### These metrics provide insights into the model’s accuracy in predicting traffic flow. Results are printed to the console after running prediction.py.