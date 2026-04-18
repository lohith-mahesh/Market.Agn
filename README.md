# Deep HFMM: Double Dueling DQN Market Maker

## Overview
This repository contains a high-frequency market making (HFMM) simulation environment and an reinforcement learning agent. The system utilizes a Double Dueling Deep Q-Network (DDDQN) to manage inventory risk and capture bid-ask spreads within a synthetic limit order book. The backend is built with FastAPI and PyTorch, serving a vanilla HTML5/JavaScript frontend for real-time telemetry.

## System Architecture

### 1. Neural Network Engine (DDDQN)
The agent utilizes a Dueling architecture to separate the estimation of the state inherent value from the advantage of specific actions.

* Value Stream V(s): Evaluates the risk of the current market state such as holding a large position during high volatility.
* Advantage Stream A(s, a): Evaluates the relative benefit of selecting a specific quote depth combination.
* Recombination: The streams are aggregated at the output layer using the formula: Q(s, a) = V(s) + (A(s, a) - mean(A(s, a'))).
* Double DQN Logic: To mitigate maximization bias, the primary network selects the optimal action while the target network performs the evaluation of that action.
* Stability Mechanisms: The implementation utilizes SmoothL1Loss (Huber Loss) to handle reward outliers and gradient clipping at a 1.0 threshold to prevent weight explosion during high-volatility events.

### 2. Market Environment Dynamics
The simulator models market microstructure mechanics to train the agent against adverse selection and toxic flow.

* Order Book Imbalance (OBI): Modeled using a discrete Ornstein-Uhlenbeck stochastic differential equation to simulate autocorrelated institutional order flow.
* State Space: A 5D continuous tensor including Normalized Net Inventory, OBI, Whale API Cooldown Ratio, Rolling 20-Tick Momentum, and Micro-Volatility. 
* State Normalization: Momentum and volatility features are scaled by a factor of 10.0 to ensure input stability for the neural network.
* Action Space: A 9D discrete space representing permutations of Bid and Ask tick depths: 1, 5, or 15 ticks from the mid-price.
* Whale Injection Mechanics: The environment supports external toxic flow injections. Slippage is calculated as (Volume / 20) * Tick Size. A 5-second hard cooldown is enforced via the API.
* Reward Shaping: Raw profit and loss is adjusted using a rolling Sortino ratio to penalize downside deviation and an Avellaneda-Stoikov quadratic inventory penalty defined as 0.5 * (Inventory / Limit)^2.

### 3. Backend and Session Management
* Framework: FastAPI serving asynchronous HTTP endpoints.
* Concurrency: Multi-tenant session isolation is handled via client-side cryptographic UUIDs.
* Compute Optimization: PyTorch is restricted to a single thread using torch.set_num_threads(1). This reduces CPU context switching latency on shared infrastructure.
* Resource Management: Environments inactive for more than 5 minutes are automatically purged from memory to prevent resource exhaustion.

## Installation and Setup

### Prerequisites
* Python 3.10+
* PyTorch
* FastAPI
* Uvicorn

### Local Deployment
1. Install dependencies:
   pip install -r requirements.txt

2. Execute the application:
   python app.py

3. Access the telemetry dashboard at http://localhost:7860
