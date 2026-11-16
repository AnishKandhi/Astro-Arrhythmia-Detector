import copy
import numpy as np
import torch
import torch.nn as nn
from train_model import CNN

test_data = np.load("unseen_test_ecg.npz")
X_time = test_data["X_time"]
X_fft = test_data["X_fft"]
y = test_data["y"]
rec_ids = test_data["rec_id"]
metas = test_data["meta"]

tr = np.load("train_ecg.npz")
np.random.seed(42)
abn_idx = np.where(tr["y"] > 0)[0]
arr_idx = np.random.choice(abn_idx, size=75, replace=False)
arr_t = tr["X_time"][arr_idx]
arr_f = tr["X_fft"][arr_idx]
arr_y = tr["y"][arr_idx]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cnn = CNN(num_classes=4).to(device)
cnn.load_state_dict(torch.load("astro_cnn.pth", map_location=device))
cnn.eval()

def evaluate_predictions(preds, targets):
    overall_acc = 100.0 * np.mean(preds == targets)
    abn_mask = targets > 0
    abn_recall = 100.0 * np.mean((preds[abn_mask] > 0)) if np.any(abn_mask) else 0.0
    norm_mask = targets == 0
    false_alarm = 100.0 * np.mean((preds[norm_mask] > 0)) if np.any(norm_mask) else 0.0
    return overall_acc, abn_recall, false_alarm

def calibrate_to_individual(pretrained_model, cal_time, cal_fft, cal_y, ev_data=None, steps=8):
    model = copy.deepcopy(pretrained_model)
    for param in model.time_branch.parameters():
        param.requires_grad = False
    for param in model.fft_branch.parameters():
        param.requires_grad = False

    mix_t = np.concatenate([cal_time, arr_t], axis=0)
    mix_f = np.concatenate([cal_fft, arr_f], axis=0)
    mix_y = np.concatenate([cal_y, arr_y], axis=0)

    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.0008)
    loss = nn.CrossEntropyLoss()

    bt = torch.tensor(mix_t).unsqueeze(1).to(device)
    bf = torch.tensor(mix_f).unsqueeze(1).to(device)
    by = torch.tensor(mix_y, dtype=torch.long).to(device)

    if ev_data is not None:
        ev_t, ev_f, ev_y = ev_data
        model.eval()
        with torch.no_grad():
            p0 = model(ev_t, ev_f).argmax(dim=1).cpu().numpy()
            acc0, rec0, fa0 = evaluate_predictions(p0, ev_y)
            print(f"  Step 00/{steps:02d} (Uncalibrated) | Acc: {acc0:6.2f}% | Sens: {rec0:6.2f}% | False Alarm: {fa0:6.2f}%")

    for step in range(1, steps + 1):
        model.train()
        optimizer.zero_grad()
        out = model(bt, bf)
        diff = loss(out, by)
        diff.backward()
        optimizer.step()

        if ev_data is not None:
            model.eval()
            with torch.no_grad():
                p = model(ev_t, ev_f).argmax(dim=1).cpu().numpy()
                acc, rec, fa = evaluate_predictions(p, ev_y)
                print(f"Step {step:02d}/{steps:02d} | Loss: {diff.item():.4f} | Acc: {acc:6.2f}% | Sens: {rec:6.2f}% | False Alarm: {fa:6.2f}%")

    model.eval()
    return model

if __name__ == "__main__":
    print("=== CROSS-DEMOGRAPHIC UNSEEN BENCHMARK REPORT ===\n")
    all_rule_preds, all_zeroshot_preds, all_personal_preds, all_targets = [], [], [], []

    for rec in np.unique(rec_ids):
        mask = rec_ids == rec
        rt, rf, ry = X_time[mask], X_fft[mask], y[mask]
        demo = metas[mask][0]
        print(f"--- Calibrating Subject {rec} ({demo}) ---")

        norms = np.where(ry == 0)[0]
        cal_idx = norms[:30]
        eval_idx = np.setdiff1d(np.arange(len(ry)), cal_idx)

        cal_t, cal_f, cal_y = rt[cal_idx], rf[cal_idx], ry[cal_idx]
        ev_t, ev_f, ev_y = rt[eval_idx], rf[eval_idx], ry[eval_idx]

        baseline_template = np.mean(cal_t, axis=0)
        dists = np.linalg.norm(ev_t - baseline_template, axis=1)
        thresh = np.mean(np.linalg.norm(cal_t - baseline_template, axis=1)) * 1.8
        rule_preds = np.where(dists > thresh, 2, 0)

        with torch.no_grad():
            bt = torch.tensor(ev_t).unsqueeze(1).to(device)
            bf = torch.tensor(ev_f).unsqueeze(1).to(device)
            zeroshot_preds = cnn(bt, bf).argmax(dim=1).cpu().numpy()

        personal_model = calibrate_to_individual(cnn, cal_t, cal_f, cal_y, ev_data=(bt, bf, ev_y), steps=30)
        with torch.no_grad():
            personal_preds = personal_model(bt, bf).argmax(dim=1).cpu().numpy()

        all_rule_preds.extend(rule_preds)
        all_zeroshot_preds.extend(zeroshot_preds)
        all_personal_preds.extend(personal_preds)
        all_targets.extend(ev_y)
        print()

    all_targets = np.array(all_targets)
    print("--- OVERALL SUMMARY ACROSS ALL UNSEEN SUBJECTS ---")
    for name, preds in [
        ("1. Rule-Based Threshold Baseline", np.array(all_rule_preds)),
        ("2. Zero-Shot Generic CNN", np.array(all_zeroshot_preds)),
        ("3. Personalized Dual-Branch CNN (Ours)", np.array(all_personal_preds)),
    ]:
        acc, rec, fa = evaluate_predictions(preds, all_targets)
        print(f"{name:40s} | Acc: {acc:6.2f}% | Arrhythmia Sensitivity: {rec:6.2f}% | False Alarm: {fa:6.2f}%")
