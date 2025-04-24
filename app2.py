import streamlit as st
import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
import matplotlib.pyplot as plt

# ——— STREAMLIT CONFIG ———
st.set_page_config(page_title="Earthquake Response Predictor", layout="wide")
st.title("🏗️ Earthquake Building Response Predictor")
st.markdown("Simulate and predict how a building will move under custom ground motion, and compare to the true response.")

# ——— MODEL & SIMULATION PARAMETERS ———
T_n = 1.0          # Natural period (s)
zeta = 0.05        # Damping ratio
omega_n = 2 * np.pi / T_n
dt = 0.01
t = np.arange(0, 10, dt)
window_size = 50   # Sequence length for LSTM

# ——— FUNCTION DEFINITIONS ———
def generate_ground_motion(t, A, alpha, omega):
    return A * np.exp(-alpha * t) * np.sin(omega * t)

def compute_displacement(a_g, t, omega_n, zeta):
    def sdof_system(x, t, a_g_func):
        x1, x2 = x
        a_g_val = a_g_func(t)
        return [x2,
                -a_g_val - 2*zeta*omega_n*x2 - omega_n**2*x1]
    a_g_func = interp1d(t, a_g, kind='linear', fill_value="extrapolate")
    x0 = [0, 0]
    sol = odeint(sdof_system, x0, t, args=(a_g_func,))
    return sol[:, 0]

def compute_Sa(a_g, t, T, zeta=0.05):
    omega_n_T = 2 * np.pi / T
    u_T = compute_displacement(a_g, t, omega_n_T, zeta)
    Sd = np.max(np.abs(u_T))
    return (2 * np.pi / T)**2 * Sd

# ——— BUILD SYNTHETIC DATABASE ———
@st.cache_resource
def generate_database(N_eq=10):
    earthquakes, features, models = [], [], []
    for _ in range(N_eq):
        A     = np.random.uniform(0.1, 1.0)
        alpha = np.random.uniform(0.1, 1.0)
        omega = np.random.uniform(1.0, 10.0)
        a_g   = generate_ground_motion(t, A, alpha, omega)
        u     = compute_displacement(a_g, t, omega_n, zeta)

        PGA    = np.max(np.abs(a_g))
        Sa_05  = compute_Sa(a_g, t, 0.5)
        Sa_10  = compute_Sa(a_g, t, 1.0)
        Sa_20  = compute_Sa(a_g, t, 2.0)
        feat   = [PGA, Sa_05, Sa_10, Sa_20]

        earthquakes.append({'a_g': a_g, 'u': u, 'feat': feat})
        features.append(feat)

        # Prepare LSTM data
        X = a_g.reshape(-1, 1)
        y = u.reshape(-1, 1)
        X_seq, y_seq = [], []
        for j in range(len(t) - window_size):
            X_seq.append(X[j : j + window_size])
            y_seq.append(y[j + window_size])
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)

        model = Sequential([
            Input(shape=(window_size, 1)),
            LSTM(50, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_seq, y_seq, epochs=10, batch_size=32, verbose=0)
        models.append(model)

    return earthquakes, np.array(features), models

earthquakes, features_array, models = generate_database()

# ——— SIDEBAR INPUTS ———
st.sidebar.header("🔧 Ground Motion Parameters")
A_new     = st.sidebar.slider("Amplitude (A)",     0.1, 1.0, 0.5)
alpha_new = st.sidebar.slider("Decay Rate (α)",    0.1, 1.0, 0.3)
omega_new = st.sidebar.slider("Frequency (ω)",     1.0, 10.0, 5.0)

# ——— NEW GROUND MOTION ———
a_g_new = generate_ground_motion(t, A_new, alpha_new, omega_new)

# ——— TRUE DISPLACEMENT ———
u_true = compute_displacement(a_g_new, t, omega_n, zeta)

# ——— FEATURE EXTRACTION & NEAREST NEIGHBOR ———
PGA_new   = np.max(np.abs(a_g_new))
Sa_05_new = compute_Sa(a_g_new, t, 0.5)
Sa_10_new = compute_Sa(a_g_new, t, 1.0)
Sa_20_new = compute_Sa(a_g_new, t, 2.0)
feat_new  = np.array([PGA_new, Sa_05_new, Sa_10_new, Sa_20_new]).reshape(1, -1)

scaler          = StandardScaler()
features_scaled = scaler.fit_transform(features_array)
feat_new_scaled = scaler.transform(feat_new)

nn = NearestNeighbors(n_neighbors=1)
nn.fit(features_scaled)
_, idx        = nn.kneighbors(feat_new_scaled)
model_similar = models[idx[0][0]]

# ——— PREDICTION WITH LSTM ———
X_new     = a_g_new.reshape(-1, 1)
X_new_seq = [X_new[j : j + window_size] for j in range(len(t) - window_size)]
X_new_seq = np.array(X_new_seq)

u_pred      = model_similar.predict(X_new_seq, verbose=0).flatten()
u_pred_full = np.zeros_like(a_g_new)
u_pred_full[window_size:] = u_pred

# ——— COMPUTE RMSE ———
rmse = np.sqrt(np.mean((u_true - u_pred_full)**2))

# ——— DISPLAY RESULTS ———
st.subheader("📊 Results")

# Show RMSE
st.markdown(f"**Root Mean Squared Error (RMSE):** {rmse:.4f} m")

# Plot true vs predicted
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t, u_true, label="True Displacement")
ax.plot(t, u_pred_full, label="Predicted Displacement", linestyle='--')
ax.set_xlabel("Time (s)")
ax.set_ylabel("Displacement (m)")
ax.set_title("True vs. Predicted Building Displacement")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Also show ground acceleration for context
st.subheader("🌀 Ground Acceleration")
st.line_chart(a_g_new)
