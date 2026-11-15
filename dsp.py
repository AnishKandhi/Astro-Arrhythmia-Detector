import wfdb
import numpy as np
from scipy.signal import butter, filtfilt

symbolMap = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,
    'A': 1, 'a': 1, 'J': 1, 'S': 1,
    'V': 2, 'E': 2,
    'F': 3, '/': 3, 'f': 3, 'Q': 3
}

def bandpass(signal, low=0.5, high=40.0, fs=360, order=4):
    b, a = butter(order, [low, high], btype='bandpass', fs=fs)
    cleaned = filtfilt(b, a, signal)
    return cleaned

def get_record(record_id, half_window=150):
    record = wfdb.rdrecord(record_id, pn_dir='mitdb')
    ann = wfdb.rdann(record_id, 'atr', pn_dir='mitdb')

    raw = record.p_signal[:, 0]
    cleaned = bandpass(raw, fs=record.fs)

    x_raw, x_time, x_fft, y = [], [], [], []
    hann = np.hanning(2 * half_window)

    for idx, symbol in zip(ann.sample, ann.symbol):
        if symbol not in symbolMap:
            continue
        if idx - half_window < 0 or idx + half_window > len(cleaned):
            continue

        raw_beat = raw[idx - half_window : idx + half_window]
        raw_beat = (raw_beat - np.mean(raw_beat)) / (np.std(raw_beat) + 1e-8)

        beat = cleaned[idx - half_window : idx + half_window]
        beat = (beat - np.mean(beat)) / (np.std(beat) + 1e-8)

        fft = np.abs(np.fft.rfft(beat * hann))
        fft = (fft - np.mean(fft)) / (np.std(fft) + 1e-8)

        x_raw.append(raw_beat)
        x_time.append(beat)
        x_fft.append(fft)
        y.append(symbolMap[symbol])

    return x_raw, x_time, x_fft, y

def build_dataset():
    np.random.seed(42)
    train_records = ['100', '102', '119', '200', '202', '208', '209', '213', '232']
    all_t, all_f, all_y = [], [], []

    for rec in train_records:
        raw, xt, xf, labels = get_record(rec)
        xt, xf, labels = np.array(xt), np.array(xf), np.array(labels)

        c0 = np.where(labels == 0)[0]
        if len(c0) > 350:
            c0 = np.random.choice(c0, size=350, replace=False)
        other = np.where(labels != 0)[0]
        keep = np.concatenate([c0, other])

        all_t.extend(xt[keep])
        all_f.extend(xf[keep])
        all_y.extend(labels[keep])
        print(f"Train Record {rec}: {len(keep)} beats")

    np.savez(
        "train_ecg.npz",
        X_time=np.array(all_t, dtype=np.float32),
        X_fft=np.array(all_f, dtype=np.float32),
        y=np.array(all_y, dtype=np.int64)
    )
    print("Saved train_ecg.npz | Class distribution:", np.bincount(all_y))

def build_unseen_profiles():
    unseen_recs = ['106', '203', '205', '107', '101', '222']
    all_raw, all_t, all_f, all_y, all_ids = [], [], [], [], []

    for rec in unseen_recs:
        xr, xt, xf, labels = get_record(rec)
        all_raw.extend(xr)
        all_t.extend(xt)
        all_f.extend(xf)
        all_y.extend(labels)
        all_ids.extend([rec] * len(labels))
        print(f"Unseen Profile {rec}: {len(labels)} beats")

    np.savez(
        "unseen_profiles.npz",
        X_raw=np.array(all_raw, dtype=np.float32),
        X_time=np.array(all_t, dtype=np.float32),
        X_fft=np.array(all_f, dtype=np.float32),
        y=np.array(all_y, dtype=np.int64),
        rec_id=np.array(all_ids)
    )
    print("Saved unseen_profiles.npz successfully")

if __name__ == "__main__":
    print("-" * 50)
    build_unseen_profiles()
