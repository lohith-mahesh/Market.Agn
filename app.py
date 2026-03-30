import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
import numpy as np
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import os

app = FastAPI()

# Payload model for incoming institutional sweeps
class WhaleAction(BaseModel):
    action: str
    volume: int

class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # Shared feature extraction layers to process market state
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU()
        )
        
        # Value stream: Evaluates how dangerous the overall market state is (V)
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(),
            nn.Linear(64, 1)
        )
        
        # Advantage stream: Evaluates the specific benefit of each quote depth (A)
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        features = self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Recombine streams, centering advantages to ensure mathematical identifiability
        qvals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return qvals

class QuantEngineEnv:
    def __init__(self):
        # Portfolio and Market baselines
        self.cash = 10000.0
        self.inv = 0
        self.mid = 100.0
        self.obi = 0.0  # Order Book Imbalance
        self.tick_sz = 0.01
        self.limit = 100
        self.internal_tick = 0
        self.mid_history = deque(maxlen=20)
        
        # State space maps to Inv, OBI, Cooldown, Momentum, Volatility
        self.state_dim = 5
        self.action_dim = 9
        self.depths = [1, 5, 15]  # Permitted quote distances from mid
        
        # Initialize primary and target networks for Double DQN stability
        self.net = DuelingDQN(self.state_dim, self.action_dim)
        self.target_net = DuelingDQN(self.state_dim, self.action_dim)
        self.target_net.load_state_dict(self.net.state_dict())
        
        # SmoothL1Loss (Huber Loss) prevents gradients from exploding during sudden price shocks
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.0005)
        self.criterion = nn.SmoothL1Loss()
        
        # Experience replay buffer and hyperparams
        self.memory = deque(maxlen=20000)
        self.batch_size = 128
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.tau = 0.005  # Soft update parameter for Polyak averaging
        
        # Telemetry tracking
        self.history = []
        self.returns = []
        self.last_pnl = 0.0
        self.whale_queue = []
        self.last_whale_time = 0.0
        self.liquidated = False
        self.last_loss = 0.0

    def trigger_whale(self, direction, volume):
        # Enforce strict 5-second API rate limit to prevent institutional spam
        if not self.liquidated:
            current_time = time.time()
            if current_time - self.last_whale_time >= 5.0:
                self.whale_queue.append((direction, volume))
                self.last_whale_time = current_time
                return True
        return False

    def _get_cooldown_ratio(self):
        # Provide the network with exact knowledge of when it can be attacked again
        elapsed = time.time() - self.last_whale_time
        return max(0.0, (5.0 - elapsed) / 5.0)

    def _get_state(self):
        momentum = (self.mid - self.mid_history[0]) if len(self.mid_history) == 20 else 0.0
        volatility = np.std(self.mid_history) if len(self.mid_history) > 5 else 0.0
        return np.array([self.inv / self.limit, self.obi, self._get_cooldown_ratio(), momentum * 10.0, volatility * 10.0], dtype=np.float32)

    def _train(self):
        if len(self.memory) < self.batch_size: return 0.0
        
        # Unpack experiences
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        
        # Current Q-values predicted by primary network
        curr_q = self.net(states).gather(1, actions).squeeze(1)
        
        # Double DQN Logic: Select best action using primary net, evaluate its value using target net
        with torch.no_grad():
            next_actions = self.net(next_states).argmax(1).unsqueeze(1)
            max_next_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            target_q = rewards + self.gamma * max_next_q * (1 - dones)
            
        loss = self.criterion(curr_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping defends against numeric instability
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.optimizer.step()
        
        # Perform soft (Polyak) updates to steadily drag target weights toward primary weights
        for target_param, local_param in zip(self.target_net.parameters(), self.net.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss.item()

    def step(self):
        if self.liquidated:
            return self._pack("-", "SYSTEM HALTED. MARGIN CALL EXECUTED.", 0, 0, 0, np.zeros(9))

        self.internal_tick += 1
        self.mid_history.append(self.mid)
        
        # Ornstein-Uhlenbeck Process for Autocorrelated Order Flow (Toxic flow clusters together)
        self.obi += 0.05 * (0.0 - self.obi) + 0.15 * np.random.normal(0, 1)
        self.obi = max(-1.0, min(1.0, self.obi))
        
        state = self._get_state()
        state_tensor = torch.tensor(state).unsqueeze(0)
        
        # Forward pass to extract Advantage Tensors for the XAI matrix
        with torch.no_grad():
            features = self.net.feature_layer(state_tensor)
            adv_tensor = self.net.advantage_stream(features).squeeze(0)
            advantages = adv_tensor.numpy()
            q_vals = self.net(state_tensor)
            best_action = torch.argmax(q_vals).item()

        # Epsilon-Greedy exploration
        if np.random.rand() < self.epsilon:
            action = np.random.randint(0, self.action_dim)
        else:
            action = best_action
                
        b_idx = action // 3
        a_idx = action % 3
        b_dep = self.depths[b_idx]
        a_dep = self.depths[a_idx]
        bid = self.mid - (b_dep * self.tick_sz)
        ask = self.mid + (a_dep * self.tick_sz)
        
        # Standard geometric Brownian motion baseline
        self.mid += np.random.normal(0.0, 0.05)
        trade_log = "-"
        
        # Process institutional sweeps
        if self.whale_queue:
            direction, vol = self.whale_queue.pop(0)
            slip = (vol // 20) * self.tick_sz
            if direction == 'buy':
                fill = ask + (slip * 0.5)
                qty = min(25, self.limit + self.inv)
                self.mid += slip # Permanent market impact
                if qty > 0:
                    self.cash += fill * qty
                    self.inv -= qty
                    trade_log = f"WHALE SWEPT ASKS (-{qty})"
            else:
                fill = bid - (slip * 0.5)
                qty = min(25, self.limit - self.inv)
                self.mid -= slip # Permanent market impact
                if qty > 0:
                    self.cash -= fill * qty
                    self.inv += qty
                    trade_log = f"WHALE SWEPT BIDS (+{qty})"
        else:
            # Process retail flow probability curve based on quote depth and OBI
            p_b = math.exp(-b_dep * 0.2) + (self.obi * 0.1)
            p_a = math.exp(-a_dep * 0.2) - (self.obi * 0.1)
            if np.random.rand() < p_a and self.inv > -self.limit:
                qty = min(np.random.randint(1, 6), self.limit + self.inv)
                if qty > 0:
                    self.cash += ask * qty
                    self.inv -= qty
                    trade_log = f"SOLD {qty} @ {ask:.2f}"
            if np.random.rand() < p_b and self.inv < self.limit:
                qty = min(np.random.randint(1, 6), self.limit - self.inv)
                if qty > 0:
                    self.cash -= bid * qty
                    self.inv += qty
                    if trade_log == "-": 
                        trade_log = f"BOUGHT {qty} @ {bid:.2f}"
                    else:
                        trade_log = "MARKET CROSSED"

        # PnL tracking
        pnl = self.cash + (self.inv * self.mid) - 10000.0
        step_ret = pnl - self.last_pnl
        self.returns.append(step_ret)
        if len(self.returns) > 100: self.returns.pop(0)
        self.last_pnl = pnl

        # Calculate Sortino ratio to aggressively penalize downside deviation
        downside = [r for r in self.returns if r < 0]
        sortino = 0.0
        if downside:
            sortino = (np.mean(self.returns) / (np.std(downside) + 1e-5)) * math.sqrt(252 * 23400)
            
        # Inventory penalty curve (Avellaneda-Stoikov logic)
        inv_penalty = 0.5 * (self.inv / self.limit)**2
        raw_reward = step_ret + (sortino * 0.05) - inv_penalty
        
        # Reward clipping enforces symmetric learning and cures directional bias
        reward = np.clip(raw_reward, -10.0, 10.0)
        
        is_done = 0.0
        if pnl <= -5000.0:
            self.liquidated = True
            reward = -10.0
            is_done = 1.0
            
        next_state = self._get_state()
        self.memory.append((state, action, reward, next_state, is_done))
        
        # Throttle backpropagation to preserve the main event loop thread
        if self.internal_tick % 5 == 0:
            loss = self._train()
            if loss > 0: self.last_loss = loss
            
        reasoning = f"Symmetric Risk Profile. | Skew: Bid {b_dep}t / Ask {a_dep}t"
        if "WHALE" in trade_log: reasoning = "STRUCTURAL SHIFT. Absorbing flow, maintaining symmetrical boundaries."
            
        return self._pack(trade_log, reasoning, b_dep, a_dep, action, advantages)

    def _pack(self, trade_log, reasoning, bd, ad, action, advantages):
        pnl = self.cash + (self.inv * self.mid) - 10000.0
        data = {
            "mid": round(self.mid, 2),
            "bid": round(self.mid - (bd * self.tick_sz), 2),
            "ask": round(self.mid + (ad * self.tick_sz), 2),
            "inv": self.inv,
            "obi": round(self.obi, 3),
            "pnl": round(pnl, 2),
            "loss": round(self.last_loss, 4), 
            "eps": round(self.epsilon, 3),
            "trade": trade_log, 
            "reasoning": reasoning,
            "liquidated": self.liquidated,
            "cooldown": round(self._get_cooldown_ratio() * 5.0, 1),
            "action": action,
            "adv": [round(float(a), 2) for a in advantages]
        }
        self.history.append(data)
        if len(self.history) > 100: self.history.pop(0)
        return data

global_env = QuantEngineEnv()

@app.get("/")
async def get_ui():
    # Absolute path resolution to dynamically find index.html next to this script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(base_dir, "index.html")
    
    if not os.path.exists(html_path):
        return HTMLResponse(f"<h1>Error: index.html not found.</h1><p>Ensure it is saved exactly at: {html_path}</p>", status_code=404)
    return FileResponse(html_path)

@app.get("/data")
async def get_data(): return {"latest": global_env.step(), "history": global_env.history}

@app.post("/whale")
async def process_whale(action: WhaleAction):
    success = global_env.trigger_whale(action.action, action.volume)
    if success: return {"status": "ok"}
    else: return {"status": "rate_limited"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)