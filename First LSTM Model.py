import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# -1. Create fake energy data (sine wave=simplified load pattern)
time_steps=1000
data = np.sin(np.linspace(0, 50, time_steps)).astype(np.float32)

#-2. Prepare squences (this is how LSTM sees time series data)
seq_length = 24           # 24 hours of history to predict next hour
X, Y = [], []
for i in range (len(data) - seq_length):
    X.append(data[i:i+seq_length])
    Y.append(data[i+seq_length])

X = torch.tensor(np.array(X)).unsqueeze(-1) #shape: (samples, 24, 1)
Y = torch.tensor(np.array(Y)).unsqueeze(-1) #shape: (samples, 1)

print("Input shape:", X.shape)
print("Target shape:", Y.shape)

# -3. Define LSTM Model
class EnergyLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super(EnergyLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) # Take the last time step
        return out

# -4. Train the model
model = EnergyLSTM()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("\nTraining...")
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, Y)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 25 == 0:
        print(f"Epoch {epoch+1}/100 | Loss: {loss.item():.6f}")
print("\nDone! Your First LSTM is trained.")

# -5. Model Evaluation through graphs

model.eval()
with torch.no_grad():
    predictions = model(X).numpy()
mse = ((predictions - Y.numpy()) **2).mean()
rmse = mse ** 0.5
mae = abs(predictions - Y.numpy()).mean()
print(f"RMSE: {rmse:.6f}")
print(f"MAE: {mae:.6f}")

plt.figure(figsize=(12, 4))
plt.plot(Y.numpy()[:100], label='Actual', color='blue')
plt.plot(predictions[:100], label='Predicted', color ='red', linestyle='--')
plt.title('Bi-LSTM: Actual vs Predicted Energy Load')
plt.xlabel('Time Steps')
plt.ylabel('Load Value')
plt.legend()
plt.tight_layout()
plt.savefig('bilstm_prediction.png')
plt.show()
