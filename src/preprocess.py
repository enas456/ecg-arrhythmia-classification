import numpy as np
import pandas as pd
from scipy.signal import firwin, filtfilt

# === Stage 1: Load and Preprocess ECG Signal ===
df = pd.read_csv(r'C:\Users\ENAS.DESKTOP-CSI0DAD\Desktop\ecg\me\dataset\100.csv')
orginal_data_ecg = df.to_numpy()

# Normalize ADC signals (assumes ch1 and ch2 in cols 1 and 2)
data_ecg_1 = (orginal_data_ecg[:, 1:3] - 1024) / 200
data_ecg_2 = np.column_stack((orginal_data_ecg[:, 0], data_ecg_1))

# Bandpass filter design (0.5–50 Hz)
fs = 360
low_freq = 0.5
high_freq = 50
nyquist_freq = fs / 2
order = 100
filter_coefficients = firwin(order + 1, [low_freq / nyquist_freq, high_freq / nyquist_freq], pass_zero='bandpass')

# Apply zero-phase filtering
filtered_ecg_1 = filtfilt(filter_coefficients, 1.0, data_ecg_2[:, 1])
filtered_ecg_2 = filtfilt(filter_coefficients, 1.0, data_ecg_2[:, 2])
new_data = np.column_stack((data_ecg_2[:, 0], filtered_ecg_1, filtered_ecg_2))

# === Stage 2: Load and Clean Annotation Data ===
data_info = pd.read_csv(r'C:\Users\ENAS.DESKTOP-CSI0DAD\Desktop\ecg\me\dataset\100annotations.txt', delimiter='\t', header=None)

# Split annotation columns
first_column_split = data_info.iloc[:, 0].astype(str).str.split(expand=True)
first_column_split = first_column_split.rename(columns={0: 'Time', 1: 'Sample_Type', 2: 'SubChanNum'})
first_column_split['FourthColumn'] = data_info.iloc[:, 1].astype(str)

# Count and replace problematic SubChanNum entries
mask = first_column_split['SubChanNum'].isin(['+', '"']) | first_column_split['SubChanNum'].isna()
num_replaced = mask.sum()

first_column_split.loc[mask, 'SubChanNum'] = first_column_split.loc[mask, 'FourthColumn']

# Convert sample index to int
first_column_split['Sample_Type'] = pd.to_numeric(first_column_split['Sample_Type'], errors='coerce')
data_info_cleaned = first_column_split.dropna(subset=['Sample_Type'])
data_info_cleaned['Sample_Type'] = data_info_cleaned['Sample_Type'].astype(int)

print(f" Annotation cleanup complete: {num_replaced} placeholder labels were replaced.")

# === Stage 3: Segment ECG Signal ===
segments = []
labels = []

for _, row in data_info_cleaned.iterrows():
    sample = row['Sample_Type']
    label = row['SubChanNum']
    start_idx = sample - 180
    end_idx = sample + 180

    segment = new_data[(new_data[:, 0] >= start_idx) & (new_data[:, 0] < end_idx), 1:]
    if segment.shape[0] == 360:
        flat_segment = segment.T.flatten()  # shape: (720,)
        segments.append(flat_segment)
        labels.append(label)

print(f" Segmentation complete: {len(segments)} valid ECG segments extracted.")

# === Stage 4: Save to Separate Excel Files ===
df_segments = pd.DataFrame(segments)
df_labels = pd.DataFrame(labels, columns=["Label"])

segments_path = r'C:\Users\ENAS.DESKTOP-CSI0DAD\Desktop\ecg\me\dataset\ecg_segments.xlsx'
labels_path = r'C:\Users\ENAS.DESKTOP-CSI0DAD\Desktop\ecg\me\dataset\ecg_labels.xlsx'

df_segments.to_excel(segments_path, index=False)
df_labels.to_excel(labels_path, index=False)

print(f" Saved segments to: {segments_path}")
print(f" Saved labels to:   {labels_path}")


import numpy as np
import pandas as pd
from scipy.signal import firwin, filtfilt
import os

# === Settings ===
input_folder = r'C:\Users\ENAS.DESKTOP-CSI0DAD\Desktop\ecg\me\dataset'
output_folder = os.path.join(input_folder, 'ECG Signal Processing and Segmentation')
os.makedirs(output_folder, exist_ok=True)  # Create folder if it doesn't exist

record_ids = list(range(100, 235))  # Adjust based on available subjects
fs = 360
low_freq = 0.5
high_freq = 50
order = 100
nyquist_freq = fs / 2
filter_coefficients = firwin(order + 1, [low_freq / nyquist_freq, high_freq / nyquist_freq], pass_zero='bandpass')

# === Loop Through Each Subject ===
for record_id in record_ids:
    try:
        ecg_file = os.path.join(input_folder, f'{record_id}.csv')
        ann_file = os.path.join(input_folder, f'{record_id}annotations.txt')

        if not os.path.exists(ecg_file) or not os.path.exists(ann_file):
            print(f" Skipping record {record_id}: files missing.")
            continue

        # Load ECG data
        df = pd.read_csv(ecg_file)
        orginal_data_ecg = df.to_numpy()
        data_ecg_1 = (orginal_data_ecg[:, 1:3] - 1024) / 200
        data_ecg_2 = np.column_stack((orginal_data_ecg[:, 0], data_ecg_1))

        # Apply filtering
        filtered_ecg_1 = filtfilt(filter_coefficients, 1.0, data_ecg_2[:, 1])
        filtered_ecg_2 = filtfilt(filter_coefficients, 1.0, data_ecg_2[:, 2])
        new_data = np.column_stack((data_ecg_2[:, 0], filtered_ecg_1, filtered_ecg_2))

        # Load and process annotation file
        data_info = pd.read_csv(ann_file, delimiter='\t', header=None)
        first_column_split = data_info.iloc[:, 0].astype(str).str.split(expand=True)
        first_column_split = first_column_split.rename(columns={0: 'Time', 1: 'Sample_Type', 2: 'SubChanNum'})
        first_column_split['FourthColumn'] = data_info.iloc[:, 1].astype(str)

        mask = first_column_split['SubChanNum'].isin(['+', '"']) | first_column_split['SubChanNum'].isna()
        first_column_split.loc[mask, 'SubChanNum'] = first_column_split.loc[mask, 'FourthColumn']

        first_column_split['Sample_Type'] = pd.to_numeric(first_column_split['Sample_Type'], errors='coerce')
        data_info_cleaned = first_column_split.dropna(subset=['Sample_Type']).copy()
        data_info_cleaned['Sample_Type'] = data_info_cleaned['Sample_Type'].astype(int)

        # Segment ECG
        segments = []
        labels = []

        for _, row in data_info_cleaned.iterrows():
            sample = row['Sample_Type']
            label = row['SubChanNum']
            start_idx = sample - 180
            end_idx = sample + 180
            segment = new_data[(new_data[:, 0] >= start_idx) & (new_data[:, 0] < end_idx), 1:]
            if segment.shape[0] == 360:
                flat_segment = segment.T.flatten()
                segments.append(flat_segment)
                labels.append(label)

        # Save to new folder with original ID names
        df_segments = pd.DataFrame(segments)
        df_labels = pd.DataFrame(labels, columns=["Label"])

        segments_path = os.path.join(output_folder, f'{record_id}_segments.xlsx')
        labels_path = os.path.join(output_folder, f'{record_id}_labels.xlsx')

        df_segments.to_excel(segments_path, index=False)
        df_labels.to_excel(labels_path, index=False)

        print(f" Saved: {record_id}_segments.xlsx and {record_id}_labels.xlsx")

    except Exception as e:
        print(f" Error processing record {record_id}: {e}")
