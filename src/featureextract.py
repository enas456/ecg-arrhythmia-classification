import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.signal import welch
import os

# === Feature Extraction Functions ===
def hjorth_parameters(sig):
    first_deriv = np.diff(sig)
    second_deriv = np.diff(first_deriv)
    var_zero = np.var(sig)
    var_d1 = np.var(first_deriv)
    var_d2 = np.var(second_deriv)
    activity = var_zero
    mobility = np.sqrt(var_d1 / var_zero) if var_zero != 0 else 0
    complexity = np.sqrt(var_d2 / var_d1) / mobility if var_d1 != 0 and mobility != 0 else 0
    return activity, mobility, complexity

def band_energy(signal, fs=360, band=(0, 10)):
    freqs, psd = welch(signal, fs=fs)
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    return np.trapz(psd[band_mask], freqs[band_mask])

def extract_ecg_features(segment_2x360):
    features = []
    for ch in segment_2x360:
        features += [
            np.mean(ch), np.std(ch), np.min(ch), np.max(ch), np.median(ch),
            skew(ch), kurtosis(ch),
            np.sum(np.abs(np.diff(np.sign(ch)))) / 2,       # Zero crossing
            np.sqrt(np.mean(ch**2)),                        # RMS
            np.sum(np.abs(ch)),                             # IEMG
            np.sum(np.abs(np.diff(ch)))                     # Waveform length
        ]
        h1, h2, h3 = hjorth_parameters(ch)
        features += [h1, h2, h3]
        features += [band_energy(ch)]
    return features

# === Paths ===
input_folder = r'C:\Users\ENAS.DESKTOP-CSI0DAD\Desktop\ecg\me\dataset\ECG Signal Processing and Segmentation'
output_folder = os.path.join(input_folder, 'features')
os.makedirs(output_folder, exist_ok=True)

# === Process All Segment Files ===
for file in os.listdir(input_folder):
    if file.endswith('_segments.xlsx'):
        record_id = file.split('_')[0]
        segment_path = os.path.join(input_folder, file)
        try:
            df_segments = pd.read_excel(segment_path)
            all_features = []

            for _, row in df_segments.iterrows():
                segment_flat = row.to_numpy()
                segment_2x360 = segment_flat.reshape(2, 360)
                features = extract_ecg_features(segment_2x360)
                all_features.append(features)

            # Save extracted features
            df_features = pd.DataFrame(all_features)
            df_features.to_excel(os.path.join(output_folder, f'{record_id}_features.xlsx'), index=False)
            print(f"✅ Features saved for: {record_id}")

        except Exception as e:
            print(f"❌ Error processing {file}: {e}")
