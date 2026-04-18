# Deep HFMM: Double Dueling DQN Market Maker

## Overview
This repository contains a high-frequency market making (HFMM) simulation environment and a reinforcement learning agent. The system utilizes a Double Dueling Deep Q-Network (DDDQN) to manage inventory risk and capture bid-ask spreads within a synthetic limit order book. The backend is built with FastAPI and PyTorch, serving a vanilla HTML5/JavaScript frontend for real-time telemetry.

## System Architecture

### 1. Neural Network Engine (DDDQN)
The agent utilizes a Dueling architecture to separate the estimation of the state inherent value from the advantage of specific actions.

* **Value Stream $V(s)$:** Evaluates the risk of the current market state.
* **Advantage Stream $A(s, a)$:** Evaluates the relative benefit of selecting specific quote depth combinations.
* **Recombination:** The streams are aggregated at the output layer using the following formula:
    $$Q(s, a) = V(s) + \left( A(s, a) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s, a') \right)$$

### 2. Double DQN Mechanism and Stability
To mitigate the maximization bias inherent in standard Q-Learning, the system implements a Double DQN logic within the training step.
* **Action Selection:** The primary network selects the optimal action.
* **Action Evaluation:** The target network provides the Q-value estimate for that selected action.
* **Optimization:** Stability is maintained via SmoothL1Loss (Huber Loss) to handle reward outliers and Gradient Clipping (norm = 1.0) to prevent weight explosion during volatility spikes.
* **Target Updates:** Employs Polyak averaging (soft updates) with $\tau = 0.005$ to transition target network weights.

### 3. Market Environment Dynamics
The simulator models market microstructure mechanics to train the agent against adverse selection:
* **Order Book Imbalance (OBI):** Modeled using a discrete Ornstein-Uhlenbeck stochastic differential equation to simulate autocorrelated institutional order flow.
* **State Space (5D Continuous):** Normalized Net Inventory, OBI, Whale API Cooldown Ratio, Rolling 20-Tick Momentum (scaled by 10.0), and Micro-Volatility (scaled by 10.0).
* **Toxic Flow Injection (Whale API):** Allows manual injection of large market orders. Slippage is calculated as $(volume // 20) \times \text{tick\_size}$. This tests the agent ability to maintain boundaries while absorbing structural shifts.
* **Reward Shaping:** Step profit/loss is adjusted using a rolling Sortino ratio and an Avellaneda-Stoikov quadratic inventory penalty: $Penalty = 0.5 \times (\frac{Inventory}{Limit})^2$.

### 4. Backend and Session Management
* **Threading Optimization:** The environment executes `torch.set_num_threads(1)` to minimize CPU context switching lag, which is critical for performance on shared-tier cloud resources.
* **Concurrency:** Multi-tenant session isolation is handled via client-side UUIDs.
* **Garbage Collection:** A 5-minute Time-To-Live (TTL) is enforced on all active environments. Inactive sessions are culled automatically to prevent memory leaks and resource exhaustion.

## Installation and Setup

### Prerequisites
* Python 3.10+

### Local Deployment
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the server:
   ```bash
   python app.py
   ```
