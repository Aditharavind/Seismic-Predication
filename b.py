import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
import matplotlib.pyplot as plt

# Define SDOF system parameters
T_n = 1.0  # Natural period (seconds)
zeta = 0.05  # Damping ratio
omega_n = 2 * np.pi / T_n

# Time array
dt = 0.01
t = np.arange(0, 10, dt)

# Generate synthetic ground motion
def generate_ground_motion(t, A, alpha, omega):
    """Generate synthetic ground acceleration time-series."""
    return A * np.exp(-alpha * t) * np.sin(omega * t)

# Compute SDOF response
def compute_displacement(a_g, t, omega_n, zeta):
    """Compute structural displacement for given ground motion."""
    def sdof_system(x, t, a_g_func):
        x1, x2 = x  # Displacement, velocity
        a_g_val = a_g_func(t)
        dx1_dt = x2
        dx2_dt = -a_g_val - 2 * zeta * omega_n * x2 - omega_n**2 * x1
        return [dx1_dt, dx2_dt]
    
    a_g_func = interp1d(t, a_g, kind='linear', fill_value="extrapolate")
    x0 = [0, 0]  # Initial conditions
    sol = odeint(sdof_system, x0, t, args=(a_g_func,))
    return sol[:, 0]  # Displacement

# Compute spectral acceleration (Sa) for a given period
def compute_Sa(a_g, t, T, zeta=0.05):
    """Compute spectral acceleration for a specific period."""
    omega_n_T = 2 * np.pi / T
    u_T = compute_displacement(a_g, t, omega_n_T, zeta)
    Sd = np.max(np.abs(u_T))  # Spectral displacement
    Sa = (2 * np.pi / T)**2 * Sd  # Spectral acceleration
    return Sa

# Build seismic information history database and SSR net
N_eq = 10  # Number of synthetic earthquakes
earthquakes = []
features = []
models = []
window_size = 50  # Sequence length for LSTM

for i in range(N_eq):
    # Random parameters for synthetic ground motion
    A = np.random.uniform(0.1, 1.0)
    alpha = np.random.uniform(0.1, 1.0)
    omega = np.random.uniform(1.0, 10.0)
    a_g = generate_ground_motion(t, A, alpha, omega)
    u = compute_displacement(a_g, t, omega_n, zeta)
    
    # Extract features for similarity (Part 1: Seismic info database)
    PGA = np.max(np.abs(a_g))
    Sa_05 = compute_Sa(a_g, t, 0.5)
    Sa_10 = compute_Sa(a_g, t, 1.0)
    Sa_20 = compute_Sa(a_g, t, 2.0)
    feat = [PGA, Sa_05, Sa_10, Sa_20]
    
    # Store data
    earthquakes.append({'a_g': a_g, 'u': u, 'feat': feat})
    features.append(feat)
    
    # Prepare sequences for LSTM (Part 2: SSR net)
    X = a_g.reshape(-1, 1)
    y = u.reshape(-1, 1)
    X_seq, y_seq = [], []
    for j in range(len(t) - window_size):
        X_seq.append(X[j:j + window_size])
        y_seq.append(y[j + window_size])
    X_seq = np.array(X_seq)  # Shape: (samples, window_size, 1)
    y_seq = np.array(y_seq)
    
    # Train individual LSTM model
    model = Sequential()
    model.add(Input(shape=(window_size, 1)))  # Define input shape here
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_seq, y_seq, epochs=10, batch_size=32, verbose=0)
    models.append(model)

# New earthquake prediction
# Generate a new synthetic earthquake
A_new = np.random.uniform(0.1, 1.0)
alpha_new = np.random.uniform(0.1, 1.0)
omega_new = np.random.uniform(1.0, 10.0)
a_g_new = generate_ground_motion(t, A_new, alpha_new, omega_new)

# Extract features for the new earthquake
PGA_new = np.max(np.abs(a_g_new))
Sa_05_new = compute_Sa(a_g_new, t, 0.5)
Sa_10_new = compute_Sa(a_g_new, t, 1.0)
Sa_20_new = compute_Sa(a_g_new, t, 2.0)
feat_new = [PGA_new, Sa_05_new, Sa_10_new, Sa_20_new]

# Part 3: Unsupervised nearest neighbor algorithm
features_array = np.array(features)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_array)
feat_new_scaled = scaler.transform([feat_new])

nn = NearestNeighbors(n_neighbors=1)
nn.fit(features_scaled)
_, idx = nn.kneighbors(feat_new_scaled)
similar_idx = idx[0][0]

# Part 4: Transfer learning - Use the model from the most similar earthquake
model_similar = models[similar_idx]

# Prepare input for prediction
X_new = a_g_new.reshape(-1, 1)
X_new_seq = []
for j in range(len(t) - window_size):
    X_new_seq.append(X_new[j:j + window_size])
X_new_seq = np.array(X_new_seq)

# Predict structural displacement
u_pred = model_similar.predict(X_new_seq, verbose=0)
u_pred_full = np.zeros_like(t)
u_pred_full[window_size:] = u_pred.flatten()

# Compute true displacement for comparison
u_true = compute_displacement(a_g_new, t, omega_n, zeta)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(t, u_true, label='True Displacement')
plt.plot(t, u_pred_full, label='Predicted Displacement')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Displacement (m)')
plt.title('Structural Seismic Response Prediction')
plt.grid(True)
plt.savefig('seismic_response_prediction.png')