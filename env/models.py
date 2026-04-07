from pydantic import BaseModel
from typing import List, Optional

class SupplyChainObservation(BaseModel):
    day: int
    inventory: int
    past_demand: List[int]
    incoming_order: int
    last_demand: Optional[int] = None

class SupplyChainAction(BaseModel):
    order_quantity: int

class SupplyChainReward(BaseModel):
    reward: float