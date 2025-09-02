# AI Negotiator Showdown 🤝

## Overview
The **AI Negotiator Showdown** is a simulation platform where multiple AI agents with unique personalities engage in automated negotiation tournaments. Each agent has a distinct negotiation strategy, and they compete to achieve the best deals across various products.  

The system integrates:
- **Rule-based strategies** (Aggressive, Diplomatic, Analytical, Wildcard).
- **Machine Learning models** (XGBoost, LSTM, SARIMA) for forecasting and decision support.
- **Tournament engine** to simulate negotiation rounds and calculate leaderboards.
- **Logging and Visualization tools** to analyze agent performance.

---

## Features
- 🤖 Multiple **AI agents** with different personalities:
  - Aggressive Trader
  - Smooth Diplomat
  - Data Analyst
  - Wildcard
- 🛒 **Products with price ranges** (Laptop, Phone, Tablet, etc.)
- 🏆 Supports **tournament types**:
  - Round Robin
  - Elimination
  - Grand Finals
- 📊 **Scoring System** based on:
  - Profit/Savings
  - Character Consistency
  - Speed Bonus
- 📈 Visualization:
  - Leaderboard charts
  - Personality analysis plots
- 📝 Detailed **logs and reports** saved in the `logs/` folder.

---

## Project Structure
```
├── agents/                # Agent definitions (Aggressive, Diplomat, Analyst, Wildcard)
├── engine/                # Tournament and negotiation engine
├── ml_models/             # Machine learning models (XGBoost, LSTM, SARIMA)
├── utils/                 # Logging and visualization utilities
├── data/                  # Dataset storage
├── logs/                  # Tournament logs and analysis outputs
├── config.py              # Global configuration (products, personalities, settings)
├── main.py                # Entry point for running tournaments
└── README.md              # Project documentation
```

---

## Installation
### Prerequisites
- Python 3.9+
- Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Usage
### Run Tournament
```bash
python main.py --tournament round_robin
```

### Train Models
```bash
python main.py --train
```

### Run with Visualizations
```bash
python main.py --tournament round_robin --visualize
```

---

## Example Output
```
📋 Generated 280 matches
🔄 Running negotiations...

🏅 TOURNAMENT RESULTS
🥇 WINNER: Aggressive_Beta (aggressive)
   Score: 1380.3
   Win Rate: 60.0%
   Avg Deal Time: 0.0s

📊 FULL LEADERBOARD:
 1. Aggressive_Beta   | 1380.3 pts | 60.0% wins | aggressive
 2. Wildcard_Hotel    | 1254.7 pts | 44.3% wins | wildcard
 ...
```

Logs and visualizations are saved in the `logs/` directory.

---

## Unique Highlights
- First project to combine **negotiation agents** + **machine learning models** + **tournament system**.
- Agents mimic **real-world personalities** (aggressive, diplomatic, analytical).
- Generates **replayable match logs** and **visual insights** for performance analysis.

---

## Team Contributions
- **Member 1** – Developed negotiation agents & personality strategies.
- **Member 2** – Built tournament engine & scoring system.
- **Member 3** – Integrated ML models (XGBoost, LSTM, SARIMA).
- **Member 4** – Created logging system & visualization dashboard.

---

## Future Work
- 🎮 Add **real-time GUI** for watching negotiations.
- 🌐 Enable **multi-agent reinforcement learning**.
- 📡 Expand product database with **real market datasets**.
- 🤝 Deploy as a **web app** for interactive experiments.

---

## License
This project is for academic/research purposes. 
