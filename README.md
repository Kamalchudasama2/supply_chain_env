# 📦 Supply Chain Decision Agent (OpenEnv)

## 🚀 Problem Description

This environment simulates a real-world inventory management problem where an agent must decide how much stock to order under uncertain demand.

The objective is to:
- Maintain sufficient inventory to meet demand
- Avoid excess stock that increases holding cost
- Minimize penalties from stockouts (unmet demand)

This reflects real-world applications such as retail inventory planning, warehouse operations, and supply chain optimization.

---

## 🧠 Observation Space

At each step, the agent receives:

```json
{
  "day": int,
  "inventory": int,
  "past_demand": [int],
  "incoming_order": int,
  "last_demand": int
}

## Description
~ day: Current timestep
~ inventory: Current stock level
~ past_demand: Recent demand history
~ incoming_order: Previously ordered stock arriving now
~ last_demand: Demand from previous step


🎮 Action Space
JSON
{
  "order_quantity": int
}

##Description
~ The agent selects how many units to order
~ Orders arrive in the next timestep
~ Large orders increase cost, small orders risk stockouts


🏆 Reward Function

The reward is defined as:

reward = - (holding_cost + stockout_penalty + order_cost)

Where:

~ Holding cost: cost of excess inventory
~ Stockout penalty: cost of unmet demand
~ Order cost: cost of placing orders

Properties:

~ Continuous reward signal at every step
~ Encourages efficient inventory management
~ Penalizes poor decisions (overstocking or understocking)
~ Normalized to range [-1.0, 1.0]

🎯 Tasks

🟢 Easy
Stable demand
Lower costs
Short horizon
Goal: Learn basic inventory balancing

🟡 Medium
Moderate demand variability
Increased penalties
Longer horizon
Goal: Balance ordering vs holding cost

🔴 Hard
High demand volatility
Strong stockout penalties
Long planning horizon
Goal: Optimize long-term decision making

⚙️ Setup Instructions

### Environment Variables (Required for inference.py)
1. Copy `.env.example` to `.env` (created below)
2. Edit `.env` with your values:
   - **API_BASE_URL**: OpenAI-compatible server URL (ex: https://your-model.hf.space/v1 from HF Inference Endpoints)
   - **HF_TOKEN**: HF token (hf_... from https://huggingface.co/settings/tokens)
   - **MODEL_NAME**: Model name (ex: meta-llama/Llama-2-7b-chat-hf)

**Quick HF Setup**:
1. https://huggingface.co/inference-endpoints → New → Select LLM + OpenAI server
2. Deploy free/paid → Copy URL/Token

1. Install dependencies
    ```
    pip install -r requirements.txt
    ```

2. **Terminal 1**: Run environment server
    ```
    uvicorn app:app --reload
    ```

3. **Terminal 2**: Test endpoints
    ```
    curl -X POST http://localhost:8000/reset
    curl -X POST http://localhost:8000/step -H "Content-Type: application/json" -d '{"order_quantity":50}'
    ```

4. **Terminal 2**: Run inference (loads .env)
    ```
    cmd /c "venv\Scripts\activate.bat && for /f "tokens=1,2 delims==" %i in (.env) do set %i=%j && python inference.py"
    ```

**Verify** (after activate & load):
```
echo %HF_TOKEN%
```

