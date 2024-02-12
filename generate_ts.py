import numpy as np
import pandas as pd

# Setting the seed for reproducibility
np.random.seed(42)

# Parameters
n_samples = 100000
n_variables = 3
timestamp = pd.date_range(start='2023-01-01', periods=n_samples, freq='T')

# Generating synthetic time-series data
data = np.empty((n_samples, n_variables))

data[:, 0] = np.sin(np.linspace(0, 20*np.pi, n_samples)) + np.random.normal(0, 0.5, n_samples)
data[:, 1] = np.linspace(0, 10, n_samples) + np.random.normal(0, 0.2, n_samples)
data[:, 2] = np.random.normal(0, 1, n_samples).cumsum()

# Converting to DataFrame
df = pd.DataFrame(data, index=timestamp, columns=['Variable1', 'Variable2', 'Variable3'])

# Introducing anomalies
anomaly_indices_spikes = np.random.choice(n_samples, size=20, replace=False)
df.iloc[anomaly_indices_spikes, np.random.randint(0, n_variables, 20)] += np.random.randint(10, 20, size=20)

# Level shift
anomaly_indices_shift = np.random.choice(n_samples, size=2, replace=False)
for shift_index in anomaly_indices_shift:
    df.iloc[shift_index:shift_index+100, np.random.randint(0, n_variables)] += 5

# Noise increase
anomaly_indices_noise = np.random.choice(n_samples, size=2, replace=False)
for noise_index in anomaly_indices_noise:
    df.iloc[noise_index:noise_index+200, np.random.randint(0, n_variables)] *= np.random.normal(1.5, 0.5, 200)

print(df.values.shape)
df.to_csv('save_csv/synthetic_ts.csv')
