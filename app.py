import copy
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.signal import stft
from train_model import CNN

st.set_page_config(page_title="Astro-Cardiac | Personalized ECG Diagnostics", layout="wide")

CLASS_INFO = {
    0: {
        "name": "Class 0: Normal Sinus Rhythm (N)",
        "severity": "success",
        "protocol": "BASELINE NOMINAL — Continue scheduled EVA / mission operations. Telemetry queued for routine downlink."
    },
    1: {
        "name": "Class 1: Supraventricular Ectopic Beat (PAC / S)",
        "severity": "warning",
        "protocol": "LOW-URGENCY ADVISORY — Early atrial beat detected (linked to dehydration, stress, or cabin CO2). Prompt crew hydration and paced breathing."
    },
    2: {
        "name": "Class 2: Ventricular Ectopic Beat (PVC / V)",
        "severity": "error",
        "protocol": "MODERATE-URGENCY ALERT — Early ventricular beat detected (linked to electrolyte shift, exertion, or oxidative stress). Pause strenuous EVA activity and replenish electrolytes."
    },
    3: {
        "name": "Class 3: Fusion / Conduction Anomaly (F / Q)",
        "severity": "error",
        "protocol": "HIGH-URGENCY CRITICAL ALERT — Compounded electrical conduction anomaly. Mandate immediate rest, run full multi-lead scan, and transmit priority telemetry to Flight Surgeon."
    }
}

PROFILES = {
    "Profile 1: Astronaut Alpha (Female, Age 24 — Record 106)": {"id": "106", "sex": "Female", "age": 24, "role": "Lunar EVA Specialist"},
    "Profile 2: Astronaut Beta (Male, Age 43 — Record 203)": {"id": "203", "sex": "Male", "age": 43, "role": "Mars Transit Commander"},
    "Profile 3: Astronaut Gamma (Male, Age 59 — Record 205)": {"id": "205", "sex": "Male", "age": 59, "role": "Orbital Flight Engineer"},
    "Profile 4: Astronaut Delta (Male, Age 63 — Record 107)": {"id": "107", "sex": "Male", "age": 63, "role": "Deep Space Payload Lead"},
    "Profile 5: Earth Patient Epsilon (Female, Age 75 — Record 101)": {"id": "101", "sex": "Female", "age": 75, "role": "Remote Rural Telehealth"},
    "Profile 6: Earth Patient Zeta (Female, Age 84 — Record 222)": {"id": "222", "sex": "Female", "age": 84, "role": "Home Cardiac Monitoring"}
}

@st.cache_resource
def load_assets():
    device = torch.device("cpu")
    base_model = CNN(num_classes=4).to(device)
    base_model.load_state_dict(torch.load("astro_cnn.pth", map_location=device))
    base_model.eval()

    test_data = np.load("unseen_profiles.npz")
    tr = np.load("train_ecg.npz")
    np.random.seed(42)
    abn_idx = np.flatnonzero(tr["y"] > 0)
    anchor_idx = np.random.choice(abn_idx, size=75, replace=False)
    anchors = (tr["X_time"][anchor_idx], tr["X_fft"][anchor_idx], tr["y"][anchor_idx])
    return device, base_model, test_data, anchors

@st.cache_resource
def calibrate_person(_base_model, rec_id, cal_t, cal_f, cal_y, anchors, steps=8):
    device = torch.device("cpu")
    model = copy.deepcopy(_base_model)
    for p in model.time_branch.parameters():
        p.requires_grad = False
    for p in model.fft_branch.parameters():
        p.requires_grad = False

    anchor_t, anchor_f, anchor_y = anchors
    mix_t = np.concatenate([cal_t, anchor_t], axis=0)
    mix_f = np.concatenate([cal_f, anchor_f], axis=0)
    mix_y = np.concatenate([cal_y, anchor_y], axis=0)

    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.0008)
    criterion = nn.CrossEntropyLoss()

    bt = torch.tensor(mix_t).unsqueeze(1).to(device)
    bf = torch.tensor(mix_f).unsqueeze(1).to(device)
    by = torch.tensor(mix_y, dtype=torch.long).to(device)

    model.train()
    for _ in range(steps):
        optimizer.zero_grad()
        loss = criterion(model(bt, bf), by)
        loss.backward()
        optimizer.step()
    model.eval()
    return model

def compute_metrics(preds, targets):
    acc = 100.0 * np.mean(preds == targets)
    abn = targets > 0
    sens = 100.0 * np.mean(preds[abn] > 0) if np.any(abn) else 0.0
    norm = targets == 0
    fa = 100.0 * np.mean(preds[norm] > 0) if np.any(norm) else 0.0
    return acc, sens, fa

@st.cache_data
def evaluate_all_individuals():
    rows = []
    for label, meta in PROFILES.items():
        rid = meta["id"]
        m = test_data["rec_id"] == rid
        rt, rf, ry = test_data["X_time"][m], test_data["X_fft"][m], test_data["y"][m]

        norm_idx = np.flatnonzero(ry == 0)
        cal_idx = norm_idx[:30]
        eval_idx = np.setdiff1d(np.arange(len(ry)), cal_idx)

        ev_t, ev_f, ev_y = rt[eval_idx], rf[eval_idx], ry[eval_idx]
        p_model = calibrate_person(base_model, rid, rt[cal_idx], rf[cal_idx], ry[cal_idx], anchors)

        with torch.no_grad():
            bt = torch.tensor(ev_t).unsqueeze(1)
            bf = torch.tensor(ev_f).unsqueeze(1)
            zs_preds = base_model(bt, bf).argmax(dim=1).numpy()
            cal_preds = p_model(bt, bf).argmax(dim=1).numpy()

        zs_acc, zs_sens, zs_fa = compute_metrics(zs_preds, ev_y)
        c_acc, c_sens, c_fa = compute_metrics(cal_preds, ev_y)

        rows.append({
            "Unseen Subject": f"Record {rid} ({meta['role']})",
            "Sex / Age": f"{meta['sex']}, {meta['age']} yrs",
            "Evaluated Beats": len(ev_y),
            "Zero-Shot Acc": f"{zs_acc:.2f}%",
            "Personalized Acc": f"{c_acc:.2f}%",
            "Arrhythmia Sensitivity": f"{c_sens:.2f}%",
            "False Alarm Rate": f"{c_fa:.2f}% (was {zs_fa:.2f}%)"
        })
    return rows

device, base_model, test_data, anchors = load_assets()

st.sidebar.title("Astro-Cardiac Mission Control")
selected_label = st.sidebar.selectbox("Select Unseen Individual (3F / 3M)", list(PROFILES.keys()))
meta = PROFILES[selected_label]
rec_id = meta["id"]

use_calibration = st.sidebar.toggle("Enable 30-Beat Personal Calibration", value=True)

mask = test_data["rec_id"] == rec_id
r_raw = test_data["X_raw"][mask]
rt = test_data["X_time"][mask]
rf = test_data["X_fft"][mask]
ry = test_data["y"][mask]

norm_indices = np.flatnonzero(ry == 0)
cal_idx = norm_indices[:30]
eval_idx = np.setdiff1d(np.arange(len(ry)), cal_idx)

ev_raw, ev_t, ev_f, ev_y = r_raw[eval_idx], rt[eval_idx], rf[eval_idx], ry[eval_idx]

active_model = (
    calibrate_person(base_model, rec_id, rt[cal_idx], rf[cal_idx], ry[cal_idx], anchors)
    if use_calibration else base_model
)

with torch.no_grad():
    all_preds = active_model(
        torch.tensor(ev_t).unsqueeze(1),
        torch.tensor(ev_f).unsqueeze(1)
    ).argmax(dim=1).numpy()

ind_acc, ind_sens, ind_fa = compute_metrics(all_preds, ev_y)

st.sidebar.markdown("---")
filter_mode = st.sidebar.radio("Inspect Beats:", ["All Beats", "Normal Beats Only", "Arrhythmia Events Only"])
if filter_mode == "Normal Beats Only":
    valid_pos = np.flatnonzero(ev_y == 0)
elif filter_mode == "Arrhythmia Events Only":
    valid_pos = np.flatnonzero(ev_y > 0)
else:
    valid_pos = np.arange(len(ev_y))

beat_slider = st.sidebar.slider("Heartbeat Index", 0, len(valid_pos) - 1, 0)
beat_idx = valid_pos[beat_slider]

st.title("Astro-Cardiac: Personalized Multi-Domain ECG Diagnostics")
st.caption(f"Active Unseen Subject: **Record {rec_id}** | **{meta['sex']}, Age {meta['age']}** ({meta['role']}) | Zero training data from this individual")

tab1, tab2 = st.tabs(["Individual Telemetry, DSP Cleaning & Beat Prediction", "6-Subject Individualized Summary"])

with tab1:
    raw_wave = ev_raw[beat_idx]
    clean_wave = ev_t[beat_idx]
    beat_fft = ev_f[beat_idx]
    true_cls = int(ev_y[beat_idx])

    with torch.no_grad():
        probs = torch.softmax(
            active_model(
                torch.tensor(clean_wave).unsqueeze(0).unsqueeze(0),
                torch.tensor(beat_fft).unsqueeze(0).unsqueeze(0)
            ),
            dim=1
        ).squeeze().numpy()
        pred_cls = int(np.argmax(probs))

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric(f"Record {rec_id} Accuracy", f"{ind_acc:.2f}%", "Personalized" if use_calibration else "Uncalibrated")
    m2.metric("Arrhythmia Sensitivity", f"{ind_sens:.2f}%")
    m3.metric("False Alarm Rate", f"{ind_fa:.2f}%")
    m4.metric("Beat Prediction", f"Class {pred_cls}", f"{probs[pred_cls]*100:.1f}% conf")
    m5.metric("Actual Ground Truth", f"Class {true_cls}", "Match" if pred_cls == true_cls else "Mismatch")

    info = CLASS_INFO[pred_cls]
    if info["severity"] == "success":
        st.success(f"**Predicted: {info['name']}** — {info['protocol']}")
    elif info["severity"] == "warning":
        st.warning(f"**Predicted: {info['name']}** — {info['protocol']}")
    else:
        st.error(f"**Predicted: {info['name']}** — {info['protocol']}")

    c_left, c_mid, c_right = st.columns(3)

    with c_left:
        st.subheader("1. DSP Signal Cleaning (Raw vs. Filtered)")
        fig1, ax1 = plt.subplots(figsize=(4.5, 3.2))
        t_axis = np.linspace(-0.417, 0.417, 300)
        ax1.plot(t_axis, raw_wave, color="#888888", alpha=0.55, linestyle="--", linewidth=1.2, label="Raw Sensor Signal")
        ax1.plot(t_axis, clean_wave, color="#00d4ff", linewidth=1.8, label="Bandpass Cleaned (0.5-40 Hz)")
        ax1.set_xlabel("Time Relative to R-Peak (s)")
        ax1.set_ylabel("Normalized Voltage")
        ax1.grid(alpha=0.25)
        ax1.legend(loc="upper right", fontsize=8)
        st.pyplot(fig1)

    with c_mid:
        st.subheader("2. 1D FFT Frequency Spectrum")
        freqs = np.fft.rfftfreq(300, d=1/360.0)
        valid_f = freqs <= 60.0
        fig2, ax2 = plt.subplots(figsize=(4.5, 3.2))
        ax2.plot(freqs[valid_f], beat_fft[valid_f], color="#ffaa00", linewidth=1.8)
        ax2.axvspan(0.5, 40.0, color="#00d4ff", alpha=0.1, label="Cardiac Passband")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Normalized FFT Magnitude")
        ax2.grid(alpha=0.25)
        ax2.legend(loc="upper right", fontsize=8)
        st.pyplot(fig2)

    with c_right:
        st.subheader("3. 2D STFT Time-Frequency Map")
        start_s = max(0, beat_idx - 3)
        end_s = min(len(ev_t), start_s + 6)
        strip = np.concatenate(ev_t[start_s:end_s])
        f_stft, t_stft, Zxx = stft(strip, fs=360, nperseg=128)
        f_mask = f_stft <= 45.0
        fig3, ax3 = plt.subplots(figsize=(4.5, 3.2))
        ax3.pcolormesh(t_stft, f_stft[f_mask], np.abs(Zxx[f_mask, :]), shading="gouraud", cmap="magma")
        ax3.set_xlabel("Multi-Beat Window (s)")
        ax3.set_ylabel("Frequency (Hz)")
        st.pyplot(fig3)
with tab2:
    st.subheader("Individualized Cross-Demographic Summary (3 Female, 3 Male — Ages 24 to 84)")
    st.write("Each unseen individual is evaluated independently on their own mission stream after 30-beat resting calibration:")
    summary_rows = evaluate_all_individuals()
    st.table(summary_rows)