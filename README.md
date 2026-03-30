# Deep HFMM: Double Dueling DQN Market Maker

![Demo](Demo.png)

## Overview

Most open-source algorithmic trading bots fail because they attempt the statistically impossible: predicting directional price movement in a noisy, chaotic system. Real proprietary trading desks do not guess direction. They act as passive liquidity providers, capturing the spread while meticulously managing inventory risk.

This repository contains the backend engine (`app.py`) for a sophisticated High-Frequency Market Making (HFMM) simulation. It implements a Double Dueling Deep Q-Network (DDDQN) that treats market making as a dynamic survival game, forcing the agent to learn asymmetric quote skewing to defend against institutional toxic order flow.

## The Mathematics and Architecture

The Python backend is not a simple wrapper; it is a custom-built market microstructure simulator and reinforcement learning environment. Here is a breakdown of the core algorithms driving the agent.

### 1. The Brain: Double Dueling Deep Q-Network (DDDQN)

A standard Deep Q-Network struggles in financial markets because it attempts to calculate a single $Q$-value for every action. In noisy environments, this leads to massive overestimation of action values. We mitigate this using a Dueling architecture.

The network splits into two independent streams after the initial feature extraction:
* **The Value Stream $V(s)$:** Calculates the inherent mathematical danger of the current market state (e.g., holding a massive long position during high volatility).
* **The Advantage Stream $A(s, a)$:** Calculates the relative benefit of selecting a specific quote depth combination.

These streams are recombined at the output layer using the following formula to ensure mathematical identifiability:
$$Q(s, a) = V(s) + \left( A(s, a) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s, a') \right)$$

To prevent the agent from developing "maximization bias" (assuming a lucky random walk was a genius strategic move), we employ **Double DQN Logic**. The primary network selects the optimal quote depth, but a completely separate target network evaluates the actual value of that decision. The target network's weights are slowly dragged toward the primary network using Polyak averaging (Soft Updates) where $\tau = 0.005$.

### 2. The Physics: Autocorrelated Order Flow (Ornstein-Uhlenbeck)

Training a bot on uniform random price action (white noise) guarantees failure in production. Institutional players use Time-Weighted Average Price (TWAP) and Iceberg algorithms, meaning heavy sell pressure usually persists over time. 

To simulate this "toxic flow clustering," the environment's Order Book Imbalance (OBI) is driven by a discrete Ornstein-Uhlenbeck stochastic differential equation. This forces the neural network to recognize sustained momentum regimes rather than reacting to instantaneous, meaningless noise.

### 3. The Critic: Reward Shaping and Risk Management

If you reward an AI strictly for total Profit and Loss (PnL), it becomes a reckless gambler. This engine shapes the reward function around risk-adjusted survival metrics.

* **Sortino Ratio Optimization:** Unlike the Sharpe ratio, which penalizes all volatility, the agent's reward is scaled by its rolling Sortino ratio. This specifically penalizes downside deviation, training the network to flatline its equity curve drawdowns.
* **Avellaneda-Stoikov Inventory Penalty:** The agent receives a continuous, asymmetric penalty based on its net exposure. The penalty scales exponentially: $Penalty \propto (\frac{Inventory}{Limit})^2$. This forces the agent to aggressively drop its Ask and widen its Bid when it gets dangerously long, naturally discovering the mechanics of inventory skewing.
* **Symmetric Reward Clipping:** To prevent the neural network from suffering permanent weight trauma from an early, unavoidable simulated crash, all raw rewards are strictly clipped between -10.0 and 10.0.

### 4. The State and Action Space

The agent evaluates the limit order book using a 5-dimensional continuous state vector:
1. Normalized Net Inventory
2. Order Book Imbalance (OU Process)
3. Cryptographic Whale Cooldown Ratio
4. Rolling 20-Tick Price Momentum
5. Micro-Volatility (Standard Deviation of price history)

Based on this tensor, the agent selects from a discrete action space of 9 possible quote depth permutations, dynamically adjusting the tick distance of its resting Bids and Asks relative to the true mid-price.

## Usage

The `app.py` script utilizes FastAPI to serve both the interactive frontend and the high-frequency telemetry endpoints. PyTorch handles the neural network matrix multiplications.

### Requirements
* Python 3.9+
* PyTorch
* FastAPI
* Uvicorn
* Numpy

### Running the Engine
Ensure `index.html` is in the same directory as `app.py`. Start the ASGI server:

```bash
pip install torch fastapi uvicorn numpy pydantic
python app.py
