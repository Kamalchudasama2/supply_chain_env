import random
from typing import Dict, Any
from .models import SupplyChainObservation, SupplyChainAction

class SupplyChainEnv:
    def __init__(self, task_config):
        self.task_config = task_config
        self.max_steps = task_config["max_steps"]
        self.reset()

    def reset(self):
        self.day = 0
        self.inventory = self.task_config["initial_inventory"]
        self.past_demand = []
        self.incoming_order = 0
        self.done = False

        return {
            "observation": SupplyChainObservation(
                day=self.day,
                inventory=self.inventory,
                past_demand=[],
                incoming_order=0
            ),
            "reward": 0.0,
            "done": False,
            "info": {}
        }

    def step(self, action: SupplyChainAction):
        if self.done:
            raise Exception("Episode done")

        self.day += 1

        # Apply incoming order
        self.inventory += self.incoming_order
        self.incoming_order = action.order_quantity

        # Generate demand
        demand = self._generate_demand()

        # Calculate unmet demand
        sales = min(self.inventory, demand)
        unmet = max(0, demand - self.inventory)

        self.inventory -= sales
        self.past_demand.append(demand)

        # Costs
        holding_cost = self.task_config["holding_cost"] * self.inventory
        stockout_cost = self.task_config["stockout_penalty"] * unmet
        order_cost = self.task_config["order_cost"] * action.order_quantity

        reward = -(holding_cost + stockout_cost + order_cost)

        # Normalize reward
        reward = max(min(reward / 100.0, 1.0), -1.0)

        if self.day >= self.max_steps:
            self.done = True

        return {
            "observation": SupplyChainObservation(
                day=self.day,
                inventory=self.inventory,
                past_demand=self.past_demand[-5:],
                incoming_order=self.incoming_order,
                last_demand=demand
            ),
            "reward": reward,
            "done": self.done,
            "info": {}
        }

    def state(self):
        return {
            "day": self.day,
            "inventory": self.inventory
        }

    def _generate_demand(self):
        base = self.task_config["demand_mean"]
        noise = random.randint(-10, 10)
        return max(0, base + noise)