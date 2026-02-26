from src.agent.graph import app
from src.surrogate.inference import predict

print("---- Feasible Viscosity Range at 75°C ----")

for f in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
    result = predict(75, f)
    print(f"Flow: {f:.2f}  →  Viscosity: {result['viscosity']:.4f}")

print("\n---- Running Closed Loop Agent ----")

initial_state = {
    "temperature": 75,
    "flow_rate": 2.5,
    "predicted_viscosity": 0.0,
    "deviation": True,
    "control_action": 0.0,
    "iteration": 0
}

result = app.invoke(initial_state)

print("\nFinal State:")
print(result)