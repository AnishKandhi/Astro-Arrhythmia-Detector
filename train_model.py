import numpy as np
import torch

import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

train_data = np.load("train_ecg.npz")
test_data = np.load("unseen_profiles.npz")

train_ds = TensorDataset(
    torch.tensor(train_data["X_time"]).unsqueeze(1),
    torch.tensor(train_data["X_fft"]).unsqueeze(1),
    torch.tensor(train_data["y"], dtype=torch.long)
)
test_ds = TensorDataset(
    torch.tensor(test_data["X_time"]).unsqueeze(1),
    torch.tensor(test_data["X_fft"]).unsqueeze(1),
    torch.tensor(test_data["y"], dtype=torch.long)
)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

class CNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.time_branch = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(4),  # 300 -> 75
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(5),  # 75 -> 15
            nn.Flatten()      # 32 * 15 = 480 
        )
        self.fft_branch = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(4),  # 151 -> 37
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),  # 37 -> 9
            nn.Flatten()      # 32 * 9 = 288 
        )
        self.classifier = nn.Sequential(
            nn.Linear(480 + 288, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x_time, x_fft):
        t_feat = self.time_branch(x_time)
        f_feat = self.fft_branch(x_fft)
        combined = torch.cat([t_feat, f_feat], dim=1)
        return self.classifier(combined)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN(num_classes=4).to(device)

    counts = np.bincount(train_data["y"])
    

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, 40):
        model.train()
        total_loss = 0.0
        for bt, bf, by in train_loader:
            bt, bf, by = bt.to(device), bf.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bt, bf)
            diff = loss(out, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for bt, bf, by in test_loader:
                bt, bf, by = bt.to(device), bf.to(device), by.to(device)
                preds = model(bt, bf).argmax(dim=1)
                correct += (preds == by).sum().item()
                total += by.size(0)

        print(f"Epoch {epoch:02d}/40 | Loss: {total_loss/len(train_loader):.4f} | Unseen Test Acc: {100*correct/total:.2f}%")

    torch.save(model.state_dict(), "astro_cnn.pth")
    print("Saved trained model to astro_cnn.pth!")
