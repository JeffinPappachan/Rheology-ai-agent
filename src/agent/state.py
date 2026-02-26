from typing import TypedDict

class PlantState(TypedDict):
    temperature: float
    flow_rate: float
    predicted_viscosity: float
    deviation: bool
    control_action: float
    iteration: int