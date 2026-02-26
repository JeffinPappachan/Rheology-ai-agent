import torch
import torch.optim as optim
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.plant.simulated_plant import SimulatedPlant
from src.surrogate.model import Surrogate
import joblib

os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

plant = SimulatedPlant()

data = []

for temp in range(60, 91, 5):
    for flow in np.linspace(0.5, 3.0, 30):
        output = plant.step(temp, 3.0, flow)
        data.append([temp, flow,
                     output["viscosity"],
                     output["shear_stress"]])

df = pd.DataFrame(data, columns=["temp", "flow", "viscosity", "shear"])
df.to_csv("data/dataset.csv", index=False)

X = df[["temp", "flow"]].values
y = df[["viscosity", "shear"]].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

joblib.dump(scaler, "models/scaler.pkl")

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

model = Surrogate()
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(300):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), "models/surrogate_model.pt")

print("Training complete. Model saved.")