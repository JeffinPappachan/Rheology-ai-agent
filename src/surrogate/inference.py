import torch
import joblib
import numpy as np
from src.surrogate.model import Surrogate

model = Surrogate()
model.load_state_dict(torch.load("models/surrogate_model.pt"))
model.eval()

scaler = joblib.load("models/scaler.pkl")

def predict(temp, flow):
    X = np.array([[temp, flow]])
    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    with torch.no_grad():
        pred = model(X_tensor).numpy()[0]

    return {
        "viscosity": float(pred[0]),
        "shear_stress": float(pred[1])
    }