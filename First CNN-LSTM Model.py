import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# -1. Generate data (same as before)
time_steps = 1000
data = np.sin(np.linspace(0, 50, time_steps)).astype(np.float32)

seq_length = 24
X, Y = [], []
for i in range(len(data) - seq_length):
    X.append(data[i:i+seq_length])
    Y.append(data[i+seq_length])

X = torch.tensor(np.array(X)).unsqueeze(-1)
Y = torch.tensor(np.array(Y)).unsqueeze(-1)
print("Input Shape:", X.shape)
print("Target Shape:", Y.shape)

# -2. CNN-BiLSTM Model
class CNN_BiLSTM(nn.Module):
    def __init__(self):
        super(CNN_BiLSTM, self).__init__()

        # CNN extracts local features from the sequence
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Bi-LSTM learns long-range temporal dependencies
        self.bilstm = nn.LSTM(input_size=64, hidden_size=64,
                              num_layers=2, batch_first=True,
                              bidirectional=True)
        
        # Output layer
        self.fc = nn.Linear(64 * 2, 1)

    def forward(self, x):
        # CNN expects (batch, channe;, sequence)
        x = x.permute(0, 2, 1)
        x = self.cnn(x)

        # Bi-LSTM expects (batch, sequence, features)
        x = x.permute(0, 2, 1)
        out, _ = self.bilstm(x)
        out = self.fc(out[:, -1, :])
        return out

# -3. Training the model
model = CNN_BiLSTM()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("\nTraining CNN-BiLSTM...")
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, Y)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 25 ==0:
        print(f"Epoch {epoch+1}/100 | Loss: {loss.item():.6f}")

print (f"Final Loss: {loss.item():.6f}")

# -4.Evaluae the model
model.eval()
with torch.no_grad():
    predictions = model(X).numpy()

mse = ((predictions - Y.numpy()) **2).mean()
rmse = mse **0.5
mae = abs(predictions - Y.numpy()).mean()
print(f"RMSE: {rmse:.6f}")
print(f"MAE: {mae:.6f}")
# -5. Plot the results
plt.figure(figsize=(12, 4))
plt.plot(Y.numpy()[:100], label="Actual", color="blue")
plt.plot(predictions[:100], label="predicted",
         color="red", linestyle="--")
plt.title("CNN-BiLSTM: Actual vs Predicted Energy Load")
plt.xlabel("Time Steps")
plt.ylabel("Load Value")
plt.legend()
plt.tight_layout()
plt.savefig("cnn_bilstm_predictions.png")
plt.show()
print("Plot Saved")
