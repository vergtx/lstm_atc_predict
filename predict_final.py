import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import joblib
import tensorflow as tf
tf.config.run_functions_eagerly(True)

# Load label encoders
label_encoders = {
    'workcode': joblib.load('workcode_label_encoder.pkl'),
    'state': joblib.load('state_label_encoder.pkl'),
    'block1': joblib.load('block1_label_encoder.pkl'),
}

# Load the trained model
model = load_model('best_model.h5')

# Debugging: Print model's input and output shape
print(f"Model input shape: {model.input_shape}")
print(f"Model output shape: {model.output_shape}")

# Define block info for max class numbers
block_info = {
    '7A': {'BAY': 39, 'ROW': 9, 'TIER': 5},
    '7B': {'BAY': 39, 'ROW': 9, 'TIER': 5},
    '7C': {'BAY': 39, 'ROW': 9, 'TIER': 5},
    '7D': {'BAY': 40, 'ROW': 9, 'TIER': 5},
    '8A': {'BAY': 34, 'ROW': 9, 'TIER': 5},
    '8B': {'BAY': 34, 'ROW': 9, 'TIER': 5},
    '8C': {'BAY': 35, 'ROW': 9, 'TIER': 5},
    '8D': {'BAY': 34, 'ROW': 9, 'TIER': 5},
}

# Preprocess the data
sample_df = pd.read_csv('tos_data_ver6_on_6th_needed_predict.csv')
for column, encoder in label_encoders.items():
    if column in sample_df.columns:
        sample_df[column] = encoder.transform(sample_df[column])

# Debugging: Print the shape and content of X_new
print(f"Shape of sample_df: {sample_df.shape}")
print(f"Content of sample_df:\n{sample_df.head()}")

# Prepare data sequences
sequence_lengths = 50
stride = sequence_lengths // 36
features = ['workcode', 'state', 'block1', 'bay1', 'row1', 'tier1', 'crane']

X_new = sample_df[features].values

# Debugging: Print the shape and content of X_new
print(f"Shape of X_new: {X_new.shape}")
print(f"Content of X_new:\n{X_new}")

X_new_sequences = []
for i in range(0, len(X_new) - sequence_lengths, stride):
    X_new_sequence = X_new[i:i + sequence_lengths]

    # Check if the input sequence is empty
    if X_new_sequence.shape[0] == 0:
        print(f"Input sequence is empty at index {i}!")
        continue  # Skip this iteration and move to the next one

    print(f"Shape of current input sequence: {X_new_sequence.shape}")

    X_new_sequences.append(X_new_sequence)

X_new_sequences = np.array(X_new_sequences)

# Debugging: Check the shape of the input sequence
print(f"Shape of input sequence: {X_new_sequences[0].shape}")

# Prediction and update
pred_bay2 = []
pred_row2 = []
pred_tier2 = []

for i in range(0, len(sample_df) - sequence_lengths, sequence_lengths):
    print(f"Shape of current input sequence: {X_new_sequences[i:i+1].shape}")
    X_sequence = X_new_sequences[i:i+1]
    print(f"Shape of current input sequence: {X_new_sequence.shape}")
    print(f"Content of current input sequence:\n{X_new_sequence}")
    
    # Check if the input sequence is empty
    if X_sequence.shape[0] == 0:
        print("Input sequence is empty!")
        break

    predictions = model.predict(X_sequence)

    # Check if the predictions are empty
    if predictions[0].size == 0 or predictions[1].size == 0 or predictions[2].size == 0:
        print("Predictions are empty!")
        break

    pred_bay2_temp, pred_row2_temp, pred_tier2_temp = predictions

     # Adjust 'pred_bay2' values by adding 1 to each prediction
    adjusted_pred_bay2 = np.argmax(pred_bay2_temp, axis=-1)[0] + 1# Prediction and update
pred_bay2 = []
pred_row2 = []
pred_tier2 = []

for i in range(0, len(sample_df) - sequence_lengths, sequence_lengths):
    print(f"Shape of current input sequence: {X_new_sequences[i:i+1].shape}")
    X_sequence = X_new_sequences[i:i+1]
    print(f"Shape of current input sequence: {X_new_sequence.shape}")
    print(f"Content of current input sequence:\n{X_new_sequence}")
    
    # Check if the input sequence is empty
    if X_sequence.shape[0] == 0:
        print("Input sequence is empty!")
        break

    predictions = model.predict(X_sequence)

    # Check if the predictions are empty
    if predictions[0].size == 0 or predictions[1].size == 0 or predictions[2].size == 0:
        print("Predictions are empty!")
        break

    pred_bay2_temp, pred_row2_temp, pred_tier2_temp = predictions
    
    # Adjust 'pred_bay2' values by adding 1 to each prediction
    adjusted_pred_bay2 = np.argmax(pred_bay2_temp, axis=-1)[0] + 1
    
    pred_bay2.extend(adjusted_pred_bay2)
    pred_row2.extend(np.argmax(pred_row2_temp, axis=-1)[0])
    pred_tier2.extend(np.argmax(pred_tier2_temp, axis=-1)[0])

# Add this code to check lengths
print(f"Length of pred_bay2: {len(pred_bay2)}")
print(f"Length of sample_df: {len(sample_df)}")

# Fill in the predictions into the original dataframe
sample_df.loc[sample_df.index[:len(pred_bay2)], 'bay2'] = pred_bay2
sample_df.loc[sample_df.index[:len(pred_row2)], 'row2'] = pred_row2
sample_df.loc[sample_df.index[:len(pred_tier2)], 'tier2'] = pred_tier2

# Save the updated dataframe
sample_df.to_csv('tos_data_ver6_on_6th_needed_predict_with_predictions.csv', index=False)
