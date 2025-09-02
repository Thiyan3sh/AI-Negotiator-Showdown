"""
enhanced_dyn_negotiator.py

Enhanced single-file dynamic negotiator with visualization:
- Generates synthetic dataset of negotiations and saves negotiation_dataset.csv
- Trains local ML models (RandomForest regressor + classifier) for buyer & seller
- Provides interactive negotiation run with live visualization
- Agents use ML models to propose offers and estimate acceptance probability
- Visual tracking of all negotiation rounds with charts
- Final results display without winner announcement

Requirements: numpy, pandas, scikit-learn, matplotlib

Run:
    pip install numpy pandas scikit-learn matplotlib
    python enhanced_dyn_negotiator.py
"""

import os
import random
import time
import math
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import seaborn as sns
import logging
from datetime import datetime

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# ---------------------------
# 1) Synthetic dataset generation
# ---------------------------
def simulate_single_negotiation(max_rounds=12):
    """Simulate a single negotiation and return list of step records."""
    market_price = random.randint(50000, 250000)
    qty = random.choice([50, 100, 200])
    grade = random.choice(["A", "B", "Export"])
    seller_min = int(market_price * random.uniform(0.65, 0.95))
    buyer_max = int(market_price * random.uniform(1.05, 1.4))

    if random.random() < 0.15:
        buyer_max = int(seller_min * random.uniform(1.0, 1.15))

    seller_style = random.choice(["aggressive", "balanced", "diplomatic"])
    buyer_style = random.choice(["aggressive", "balanced", "diplomatic"])

    seller_offer = int(market_price * {"aggressive": 1.25, "balanced": 1.12, "diplomatic": 1.08}[seller_style])
    buyer_offer = int(market_price * {"aggressive": 0.78, "balanced": 0.90, "diplomatic": 0.96}[buyer_style])

    seller_offer = max(seller_offer, seller_min)
    buyer_offer = min(buyer_offer, buyer_max)

    history = []
    records = []
    turn = 0
    for r in range(max_rounds):
        time_left = max_rounds - r
        if turn % 2 == 0:
            speaker = "seller"
            own_limit = seller_min
            opponent_limit = buyer_max
            current_offer = seller_offer
            persona = seller_style
        else:
            speaker = "buyer"
            own_limit = buyer_max
            opponent_limit = seller_min
            current_offer = buyer_offer
            persona = buyer_style

        if persona == "aggressive":
            concession_factor = random.uniform(0.01, 0.035)
        elif persona == "balanced":
            concession_factor = random.uniform(0.03, 0.07)
        else:
            concession_factor = random.uniform(0.06, 0.12)

        if speaker == "seller":
            next_offer = int(max(own_limit, current_offer - max(1000, concession_factor * current_offer)))
        else:
            next_offer = int(min(own_limit, current_offer + max(1000, concession_factor * current_offer)))

        opponent_last_amount = history[-1]["amount"] if history else None
        accept_prob = 0.0
        if opponent_last_amount is not None:
            if speaker == "seller":
                gap = (opponent_last_amount - own_limit) / max(1, own_limit)
            else:
                gap = (own_limit - opponent_last_amount) / max(1, own_limit)
            accept_prob = 1 / (1 + math.exp(-15 * (gap - 0.02)))
            if persona == "aggressive":
                accept_prob *= 0.6
            elif persona == "diplomatic":
                accept_prob = min(1.0, accept_prob * 1.2)
            accept_prob = min(1.0, accept_prob + 0.4 * (r / max_rounds))

        accepted = 0
        if opponent_last_amount is not None and random.random() < accept_prob:
            accepted = 1

        rec = {
            "role": speaker,
            "market_price": market_price,
            "qty": qty,
            "grade": grade,
            "seller_min": seller_min,
            "buyer_max": buyer_max,
            "persona": persona,
            "round": r,
            "time_left": time_left,
            "last_own_offer": current_offer,
            "opponent_last_offer": opponent_last_amount if opponent_last_amount is not None else -1,
            "next_offer": next_offer,
            "opponent_offer_accepted": accepted
        }
        records.append(rec)

        history.append({"who": speaker, "amount": current_offer})
        if speaker == "seller":
            seller_offer = next_offer
        else:
            buyer_offer = next_offer

        if accepted == 1:
            break

        turn += 1

    return records

def generate_dataset(n_negotiations=4000):
    all_records = []
    for _ in range(n_negotiations):
        recs = simulate_single_negotiation(max_rounds=random.randint(6, 14))
        all_records.extend(recs)
    df = pd.DataFrame(all_records)
    df['grade_code'] = df['grade'].map({'A': 2, 'B': 1, 'Export': 3})
    df['role_code'] = df['role'].map({'seller': 0, 'buyer': 1})
    df['persona_code'] = df['persona'].map({'aggressive': 0, 'balanced': 1, 'diplomatic': 2})
    df['opponent_last_offer'] = df['opponent_last_offer'].fillna(-1)
    df.to_csv("negotiation_dataset.csv", index=False)
    print("Saved synthetic dataset to negotiation_dataset.csv; rows:", len(df))
    return df

# ---------------------------
# 2) Train ML models
# ---------------------------
def train_models(df: pd.DataFrame):
    features = [
        'market_price', 'qty', 'grade_code', 'seller_min', 'buyer_max',
        'persona_code', 'round', 'time_left', 'last_own_offer', 'opponent_last_offer', 'role_code'
    ]
    X = df[features].values
    y_offer = df['next_offer'].values
    y_accept = df['opponent_offer_accepted'].values

    X_train, X_test, y_offer_train, y_offer_test, y_accept_train, y_accept_test = train_test_split(
        X, y_offer, y_accept, test_size=0.15, random_state=RANDOM_SEED
    )

    reg = RandomForestRegressor(n_estimators=120, random_state=RANDOM_SEED, n_jobs=-1)
    reg.fit(X_train, y_offer_train)
    pred_offer = reg.predict(X_test)
    mae = mean_absolute_error(y_offer_test, pred_offer)
    print(f"Trained offer regressor. MAE on test: {mae:.1f}")

    clf = RandomForestClassifier(n_estimators=120, random_state=RANDOM_SEED, n_jobs=-1)
    clf.fit(X_train, y_accept_train)
    pred_accept = clf.predict(X_test)
    acc = accuracy_score(y_accept_test, pred_accept)
    print(f"Trained accept classifier. Accuracy on test: {acc:.3f}")

    return {
        "offer_model": reg,
        "accept_model": clf,
        "features": features
    }

# ---------------------------
# 3) Logging Setup
# ---------------------------
def setup_negotiation_logger(product_name):
    """Setup logger for detailed negotiation tracking"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"negotiation_log_{timestamp.replace(':', '-')}.txt"
    
    # Create logger
    logger = logging.getLogger('negotiation')
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create file handler
    file_handler = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(file_handler)
    
    # Log initial header
    logger.info("="*80)
    logger.info("DYNAMIC NEGOTIATION SESSION LOG")
    logger.info("="*80)
    logger.info(f"Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Product: {product_name}")
    logger.info("="*80)
    
    print(f"Negotiation log will be saved as: {log_filename}")
    return logger, log_filename

# ---------------------------
# 4) Visualization Classes
# ---------------------------
class NegotiationVisualizer:
    def __init__(self, market_price, seller_min, buyer_max):
        self.market_price = market_price
        self.seller_min = seller_min
        self.buyer_max = buyer_max
        self.negotiation_history = []
        self.accept_probs = []
        
        # Setup the plot
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 10))
        self.fig.suptitle('Dynamic Negotiation Tracker', fontsize=16, fontweight='bold')
        
        # Initialize plots
        self.setup_plots()
        plt.ion()  # Interactive mode
        plt.show(block=False)
        
    def setup_plots(self):
        # Plot 1: Offer progression
        self.ax1.set_title('Offer Progression Throughout Negotiation', fontweight='bold')
        self.ax1.set_xlabel('Round')
        self.ax1.set_ylabel('Price (₹)')
        
        # Add reference lines
        self.ax1.axhline(y=self.market_price, color='gray', linestyle='--', alpha=0.7, label='Market Price')
        self.ax1.axhline(y=self.seller_min, color='red', linestyle=':', alpha=0.7, label='Seller Minimum')
        self.ax1.axhline(y=self.buyer_max, color='blue', linestyle=':', alpha=0.7, label='Buyer Maximum')
        
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)
        
        # Plot 2: Acceptance probability
        self.ax2.set_title('Acceptance Probability by Round', fontweight='bold')
        self.ax2.set_xlabel('Round')
        self.ax2.set_ylabel('Acceptance Probability')
        self.ax2.set_ylim(0, 1)
        self.ax2.grid(True, alpha=0.3)
        
    def update_visualization(self, round_no, role, offer, accept_prob=None, accepted=False):
        """Update the visualization with new round data"""
        self.negotiation_history.append({
            'round': round_no,
            'role': role,
            'offer': offer,
            'accepted': accepted
        })
        
        if accept_prob is not None:
            self.accept_probs.append({
                'round': round_no,
                'prob': accept_prob,
                'role': role
            })
        
        self.redraw_plots()
        
    def redraw_plots(self):
        """Redraw both plots with current data"""
        # Clear and redraw plot 1
        self.ax1.clear()
        self.setup_plots()
        
        # Plot offers
        seller_rounds = [h['round'] for h in self.negotiation_history if h['role'] == 'seller']
        seller_offers = [h['offer'] for h in self.negotiation_history if h['role'] == 'seller']
        buyer_rounds = [h['round'] for h in self.negotiation_history if h['role'] == 'buyer']
        buyer_offers = [h['offer'] for h in self.negotiation_history if h['role'] == 'buyer']
        
        if seller_rounds:
            self.ax1.plot(seller_rounds, seller_offers, 'ro-', linewidth=2, 
                         markersize=8, label='Seller Offers', color='#ff6b6b')
        if buyer_rounds:
            self.ax1.plot(buyer_rounds, buyer_offers, 'bo-', linewidth=2, 
                         markersize=8, label='Buyer Offers', color='#4ecdc4')
        
        # Highlight accepted offer if any
        accepted_offers = [h for h in self.negotiation_history if h['accepted']]
        if accepted_offers:
            final_offer = accepted_offers[-1]
            self.ax1.plot(final_offer['round'], final_offer['offer'], 'go', 
                         markersize=15, label='DEAL CLOSED!', color='#2ecc71')
        
        self.ax1.legend()
        
        # Update plot 2
        self.ax2.clear()
        self.ax2.set_title('Acceptance Probability by Round', fontweight='bold')
        self.ax2.set_xlabel('Round')
        self.ax2.set_ylabel('Acceptance Probability')
        self.ax2.set_ylim(0, 1)
        self.ax2.grid(True, alpha=0.3)
        
        if self.accept_probs:
            rounds = [p['round'] for p in self.accept_probs]
            probs = [p['prob'] for p in self.accept_probs]
            roles = [p['role'] for p in self.accept_probs]
            
            # Color by role
            colors = ['#ff6b6b' if role == 'seller' else '#4ecdc4' for role in roles]
            self.ax2.scatter(rounds, probs, c=colors, s=60, alpha=0.7)
            
            if len(rounds) > 1:
                self.ax2.plot(rounds, probs, 'k--', alpha=0.5)
        
        # Add acceptance threshold line
        self.ax2.axhline(y=0.6, color='orange', linestyle='--', alpha=0.7, 
                        label='Acceptance Threshold (0.6)')
        self.ax2.legend()
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)
        
    def show_final_summary(self, deal_price, deal_round, seller_profit, buyer_savings, 
                          seller_score, buyer_score):
        """Display final negotiation summary"""
        # Clear the second subplot for summary
        self.ax2.clear()
        self.ax2.set_xlim(0, 10)
        self.ax2.set_ylim(0, 10)
        self.ax2.axis('off')
        
        # Create summary text
        if deal_price is not None:
            summary_text = f"""
NEGOTIATION COMPLETED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Deal Price: ₹{deal_price:,}
Closed at Round: {deal_round}

FINANCIAL OUTCOMES:
• Seller profit over minimum: ₹{seller_profit:,}
• Buyer savings under maximum: ₹{buyer_savings:,}

PERFORMANCE SCORES:
• Seller combined score: {seller_score:.3f}
• Buyer combined score: {buyer_score:.3f}

Market Price Reference: ₹{self.market_price:,}
            """
        else:
            summary_text = f"""
NEGOTIATION COMPLETED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Result: NO DEAL REACHED

Both parties failed to reach agreement
within the maximum number of rounds.

PERFORMANCE SCORES:
• Seller score: 0.000
• Buyer score: 0.000

Market Price Reference: ₹{self.market_price:,}
            """
        
        self.ax2.text(0.5, 5, summary_text, fontsize=11, ha='center', va='center',
                     bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        plt.draw()
        
        # Keep the plot open for viewing
        input("\nPress Enter to close the visualization...")
        plt.close()

# ---------------------------
# 5) Agent wrappers
# ---------------------------
class MLAgent:
    def __init__(self, role: str, models: Dict[str, Any], feature_list: List[str], persona_hint: str, logger=None):
        self.role = role
        self.models = models
        self.features = feature_list
        self.persona_hint = persona_hint
        self.history = []
        self.own_limit = None
        self.market_price = None
        self.qty = None
        self.grade_code = None
        self.logger = logger

    def set_context(self, market_price:int, qty:int, grade_code:int, own_limit:int, opp_limit:int):
        self.market_price = market_price
        self.qty = qty
        self.grade_code = grade_code
        self.own_limit = own_limit
        self.opp_limit = opp_limit
        
        # Log agent setup
        if self.logger:
            self.logger.info(f"\n{self.role.upper()} AGENT INITIALIZED:")
            self.logger.info(f"  • Persona: {self.persona_hint}")
            self.logger.info(f"  • Own limit: ₹{own_limit:,}")
            self.logger.info(f"  • Opponent limit: ₹{opp_limit:,}")
            self.logger.info(f"  • Market price reference: ₹{market_price:,}")

    def make_features(self, round_no:int, time_left:int, last_own_offer:int, opponent_last_offer:int):
        feat_map = {
            'market_price': self.market_price,
            'qty': self.qty,
            'grade_code': self.grade_code,
            'seller_min': self.own_limit if self.role=='seller' else self.opp_limit,
            'buyer_max': self.own_limit if self.role=='buyer' else self.opp_limit,
            'persona_code': {'aggressive':0,'balanced':1,'diplomatic':2}.get(self.persona_hint,1),
            'round': round_no,
            'time_left': time_left,
            'last_own_offer': last_own_offer if last_own_offer is not None else -1,
            'opponent_last_offer': opponent_last_offer if opponent_last_offer is not None else -1,
            'role_code': 0 if self.role=='seller' else 1
        }
        return np.array([feat_map[f] for f in self.features]).reshape(1, -1)

    def propose(self, round_no:int, time_left:int, last_own_offer:int, opponent_last_offer:int) -> int:
        X = self.make_features(round_no, time_left, last_own_offer or -1, opponent_last_offer or -1)
        pred = self.models['offer_model'].predict(X)[0]
        if self.role == 'seller':
            pred = max(int(pred), int(self.own_limit))
        else:
            pred = min(int(pred), int(self.own_limit))
        jitter = int(max(500, abs(pred)*0.015))
        pred = int(pred + random.randint(-jitter, jitter))
        if self.role == 'seller':
            pred = max(pred, int(self.own_limit))
        else:
            pred = min(pred, int(self.own_limit))
        
        # Log detailed proposal reasoning
        if self.logger:
            self.logger.info(f"\n{self.role.upper()} PROPOSAL ANALYSIS (Round {round_no}):")
            self.logger.info(f"  • ML Model predicted: ₹{int(self.models['offer_model'].predict(X)[0]):,}")
            self.logger.info(f"  • After limit constraint: ₹{int(max(int(self.models['offer_model'].predict(X)[0]), int(self.own_limit)) if self.role=='seller' else min(int(self.models['offer_model'].predict(X)[0]), int(self.own_limit))):,}")
            self.logger.info(f"  • Jitter applied: ±₹{jitter:,}")
            self.logger.info(f"  • Final proposal: ₹{pred:,}")
            self.logger.info(f"  • Distance from own limit: ₹{abs(pred - self.own_limit):,}")
            if opponent_last_offer and opponent_last_offer != -1:
                gap = abs(pred - opponent_last_offer)
                self.logger.info(f"  • Gap from opponent's last offer: ₹{gap:,}")
            self.logger.info(f"  • Rounds remaining: {time_left}")
        
        self.history.append({'round': round_no, 'proposal': pred, 'time_left': time_left})
        return int(pred)

    def evaluate_acceptance(self, round_no:int, time_left:int, last_own_offer:int, opponent_offer:int) -> float:
        X = self.make_features(round_no, time_left, last_own_offer or -1, opponent_offer or -1)
        base_prob = self.models['accept_model'].predict_proba(X)[0][1]
        prob = min(1.0, base_prob + 0.25 * (1 - time_left/15.0))
        
        # Log detailed acceptance analysis
        if self.logger:
            self.logger.info(f"\n{self.role.upper()} ACCEPTANCE EVALUATION:")
            self.logger.info(f"  • Opponent offer: ₹{opponent_offer:,}")
            self.logger.info(f"  • Base ML probability: {base_prob:.3f}")
            self.logger.info(f"  • Time pressure bonus: {0.25 * (1 - time_left/15.0):.3f}")
            self.logger.info(f"  • Final acceptance probability: {prob:.3f}")
            
            # Calculate financial impact
            if self.role == 'seller':
                profit = opponent_offer - self.own_limit
                self.logger.info(f"  • Potential profit: ₹{profit:,}")
                self.logger.info(f"  • Profit margin: {(profit/max(1,self.own_limit)*100):.1f}%")
            else:
                savings = self.own_limit - opponent_offer  
                self.logger.info(f"  • Potential savings: ₹{savings:,}")
                self.logger.info(f"  • Savings percentage: {(savings/max(1,self.own_limit)*100):.1f}%")
        
        return float(prob)

# ---------------------------
# 6) Enhanced negotiation engine with visualization and logging
# ---------------------------
def run_negotiation_session(models, features):
    print("\n=== Enhanced Dynamic Negotiator with Visualization & Logging ===")
    print("Available products (examples):")
    products = [
        {"name":"100 boxes Grade-A Alphonso mangoes","market_price":175000,"qty":100,"grade_code":2},
        {"name":"50 bags Premium coffee beans","market_price":90000,"qty":50,"grade_code":1},
        {"name":"200 stems Exotic flowers (seasonal)","market_price":120000,"qty":200,"grade_code":2},
        {"name":"100 boxes Kesar mangoes","market_price":135000,"qty":100,"grade_code":1}
    ]
    for i,p in enumerate(products):
        print(f"{i+1}. {p['name']} - market reference: ₹{p['market_price']:,}")

    selection = input("Select product number (1-4) or press Enter for 1: ").strip()
    if selection == "":
        selection = "1"
    sel = products[int(selection)-1]
    market_price = sel['market_price']
    qty = sel['qty']
    grade_code = sel['grade_code']

    buyer_max = int(input(f"Enter BUYER max budget (total) [suggested ~ {int(market_price*1.2)}]: ") or int(market_price*1.2))
    seller_min = int(input(f"Enter SELLER min acceptable (total) [suggested ~ {int(market_price*0.8)}]: ") or int(market_price*0.8))

    seller_persona = input("Seller persona (aggressive/balanced/diplomatic) [balanced]: ") or "balanced"
    buyer_persona = input("Buyer persona (aggressive/balanced/diplomatic) [balanced]: ") or "balanced"

    # Setup logging
    logger, log_filename = setup_negotiation_logger(sel['name'])
    
    # Log negotiation parameters
    logger.info(f"\nNEGOTIATION PARAMETERS:")
    logger.info(f"  • Market Price: ₹{market_price:,}")
    logger.info(f"  • Quantity: {qty}")
    logger.info(f"  • Grade: {sel['name'].split()[-1] if 'Grade' in sel['name'] else 'Standard'}")
    logger.info(f"  • Buyer Maximum: ₹{buyer_max:,}")
    logger.info(f"  • Seller Minimum: ₹{seller_min:,}")
    logger.info(f"  • BATNA (Best Alternative): Market Price ₹{market_price:,}")
    logger.info(f"  • Negotiation Zone: ₹{buyer_max - seller_min:,} (overlap: {'Yes' if buyer_max >= seller_min else 'No'})")

    # Initialize visualization
    visualizer = NegotiationVisualizer(market_price, seller_min, buyer_max)
    
    seller_agent = MLAgent(role='seller', models=models, feature_list=features, persona_hint=seller_persona, logger=logger)
    buyer_agent  = MLAgent(role='buyer', models=models, feature_list=features, persona_hint=buyer_persona, logger=logger)

    seller_agent.set_context(market_price, qty, grade_code, own_limit=seller_min, opp_limit=buyer_max)
    buyer_agent.set_context(market_price, qty, grade_code, own_limit=buyer_max, opp_limit=seller_min)

    print(f"\nStarting negotiation for: {sel['name']}")
    print(f"Market reference: ₹{market_price:,}")
    print(f"Seller minimum: ₹{seller_min:,}, Buyer maximum: ₹{buyer_max:,}")
    print("Visualization window should now be open. Check your screen!")
    print(f"Detailed log will be saved to: {log_filename}\n")
    
    # Log negotiation start
    logger.info(f"\n" + "="*60)
    logger.info("NEGOTIATION ROUNDS BEGIN")
    logger.info("="*60)
    
    time.sleep(2)

    round_no = 0
    max_rounds = 18
    init_seller_offer = int(market_price * {"aggressive":1.25,"balanced":1.12,"diplomatic":1.08}.get(seller_persona,1.12))
    init_buyer_offer  = int(market_price * {"aggressive":0.78,"balanced":0.90,"diplomatic":0.96}.get(buyer_persona,0.90))
    init_seller_offer = max(init_seller_offer, seller_min)
    init_buyer_offer  = min(init_buyer_offer, buyer_max)

    last_seller_offer = init_seller_offer
    last_buyer_offer = init_buyer_offer

    turn = 0
    deal_price = None
    deal_round = None

    while round_no < max_rounds:
        time_left = max_rounds - round_no
        
        # Log round header
        logger.info(f"\n{'='*40}")
        logger.info(f"ROUND {round_no} (Time Left: {time_left} rounds)")
        logger.info(f"{'='*40}")
        
        if turn == 0:
            proposal = seller_agent.propose(round_no, time_left, last_seller_offer, last_buyer_offer)
            print(f"Round {round_no} - Seller proposes: ₹{proposal:,}")
            
            # Log round summary
            logger.info(f"\nROUND SUMMARY:")
            logger.info(f"  • Seller proposed: ₹{proposal:,}")
            
            # Update visualization
            visualizer.update_visualization(round_no, 'seller', proposal)
            
            accept_prob = buyer_agent.evaluate_acceptance(round_no, time_left, last_buyer_offer, proposal)
            
            # Update visualization with acceptance probability
            visualizer.accept_probs.append({
                'round': round_no,
                'prob': accept_prob,
                'role': 'buyer'  # buyer is evaluating seller's offer
            })
            visualizer.redraw_plots()
            
            if proposal <= buyer_agent.own_limit or accept_prob > 0.6:
                print(f"Buyer ACCEPTS at ₹{proposal:,} (accept_prob={accept_prob:.2f})")
                logger.info(f"\n🎉 BUYER ACCEPTS THE OFFER!")
                logger.info(f"  • Accepted price: ₹{proposal:,}")
                logger.info(f"  • Acceptance probability was: {accept_prob:.3f}")
                logger.info(f"  • Decision factors:")
                if proposal <= buyer_agent.own_limit:
                    logger.info(f"    - Offer within budget (≤₹{buyer_agent.own_limit:,})")
                if accept_prob > 0.6:
                    logger.info(f"    - High acceptance probability ({accept_prob:.3f} > 0.6)")
                
                deal_price = proposal
                deal_round = round_no
                # Mark as accepted in visualization
                visualizer.negotiation_history[-1]['accepted'] = True
                visualizer.redraw_plots()
                break
            else:
                print(f"Buyer rejects (accept_prob={accept_prob:.2f})")
                logger.info(f"\n❌ BUYER REJECTS THE OFFER")
                logger.info(f"  • Rejection reasons:")
                if proposal > buyer_agent.own_limit:
                    logger.info(f"    - Exceeds budget (₹{proposal:,} > ₹{buyer_agent.own_limit:,})")
                if accept_prob <= 0.6:
                    logger.info(f"    - Low acceptance probability ({accept_prob:.3f} ≤ 0.6)")
                
            last_seller_offer = proposal
        else:
            proposal = buyer_agent.propose(round_no, time_left, last_buyer_offer, last_seller_offer)
            print(f"Round {round_no} - Buyer proposes: ₹{proposal:,}")
            
            # Log round summary
            logger.info(f"\nROUND SUMMARY:")
            logger.info(f"  • Buyer proposed: ₹{proposal:,}")
            
            # Update visualization
            visualizer.update_visualization(round_no, 'buyer', proposal)
            
            accept_prob = seller_agent.evaluate_acceptance(round_no, time_left, last_seller_offer, proposal)
            
            # Update visualization with acceptance probability
            visualizer.accept_probs.append({
                'round': round_no,
                'prob': accept_prob,
                'role': 'seller'  # seller is evaluating buyer's offer
            })
            visualizer.redraw_plots()
            
            if proposal >= seller_agent.own_limit or accept_prob > 0.6:
                print(f"Seller ACCEPTS at ₹{proposal:,} (accept_prob={accept_prob:.2f})")
                logger.info(f"\n🎉 SELLER ACCEPTS THE OFFER!")
                logger.info(f"  • Accepted price: ₹{proposal:,}")
                logger.info(f"  • Acceptance probability was: {accept_prob:.3f}")
                logger.info(f"  • Decision factors:")
                if proposal >= seller_agent.own_limit:
                    logger.info(f"    - Offer meets minimum (≥₹{seller_agent.own_limit:,})")
                if accept_prob > 0.6:
                    logger.info(f"    - High acceptance probability ({accept_prob:.3f} > 0.6)")
                
                deal_price = proposal
                deal_round = round_no
                # Mark as accepted in visualization
                visualizer.negotiation_history[-1]['accepted'] = True
                visualizer.redraw_plots()
                break
            else:
                print(f"Seller rejects (accept_prob={accept_prob:.2f})")
                logger.info(f"\n❌ SELLER REJECTS THE OFFER")
                logger.info(f"  • Rejection reasons:")
                if proposal < seller_agent.own_limit:
                    logger.info(f"    - Below minimum (₹{proposal:,} < ₹{seller_agent.own_limit:,})")
                if accept_prob <= 0.6:
                    logger.info(f"    - Low acceptance probability ({accept_prob:.3f} ≤ 0.6)")
                
            last_buyer_offer = proposal

        # Log current negotiation gap
        current_gap = abs(last_seller_offer - last_buyer_offer)
        logger.info(f"\nCURRENT NEGOTIATION STATE:")
        logger.info(f"  • Seller's last offer: ₹{last_seller_offer:,}")
        logger.info(f"  • Buyer's last offer: ₹{last_buyer_offer:,}")
        logger.info(f"  • Gap between positions: ₹{current_gap:,}")
        logger.info(f"  • Market price reference: ₹{market_price:,}")

        turn = 1 - turn
        round_no += 1
        time.sleep(1)  # Slower pace for better visualization

    # Log final results
    logger.info(f"\n" + "="*60)
    logger.info("NEGOTIATION CONCLUDED")
    logger.info("="*60)
    
    # Calculate final scores and display results
    if deal_price is None:
        print("\nNo deal reached within maximum rounds.")
        logger.info(f"\n💔 NO DEAL REACHED")
        logger.info(f"  • Rounds exhausted: {max_rounds}")
        logger.info(f"  • Final seller position: ₹{last_seller_offer:,}")
        logger.info(f"  • Final buyer position: ₹{last_buyer_offer:,}")
        logger.info(f"  • Final gap: ₹{abs(last_seller_offer - last_buyer_offer):,}")
        logger.info(f"  • Both parties score: 0.000")
        
        seller_profit = 0
        buyer_savings = 0
        seller_final = 0.0
        buyer_final = 0.0
    else:
        seller_profit = deal_price - seller_min
        buyer_savings = buyer_max - deal_price
        print(f"\n=== DEAL COMPLETED ===")
        print(f"Final price: ₹{deal_price:,} (Round {deal_round})")
        print(f"Seller profit over minimum: ₹{seller_profit:,}")
        print(f"Buyer savings under maximum: ₹{buyer_savings:,}")

        # Log successful deal details
        logger.info(f"\n✅ SUCCESSFUL DEAL!")
        logger.info(f"  • Final price: ₹{deal_price:,}")
        logger.info(f"  • Completed in round: {deal_round}")
        logger.info(f"  • Negotiation efficiency: {((max_rounds - deal_round) / max_rounds * 100):.1f}%")
        
        logger.info(f"\nFINANCIAL OUTCOMES:")
        logger.info(f"  • Seller profit over minimum: ₹{seller_profit:,} ({(seller_profit/seller_min*100):.1f}%)")
        logger.info(f"  • Buyer savings under maximum: ₹{buyer_savings:,} ({(buyer_savings/buyer_max*100):.1f}%)")
        logger.info(f"  • Deal vs Market Price: {'₹{:,} above'.format(deal_price - market_price) if deal_price > market_price else '₹{:,} below'.format(market_price - deal_price)} market")

        seller_char = 0.5 + 0.02 * len(seller_agent.history)
        buyer_char  = 0.5 + 0.02 * len(buyer_agent.history)
        speed_score = max(0, (max_rounds - deal_round) / max_rounds)

        seller_final = 0.4 * min(1, seller_profit / max(1, buyer_max - seller_min)) + 0.4 * min(1, seller_char) + 0.2 * speed_score
        buyer_final  = 0.4 * min(1, buyer_savings / max(1, buyer_max - seller_min)) + 0.4 * min(1, buyer_char) + 0.2 * speed_score

        print(f"\nPerformance Scores:")
        print(f"Seller combined score: {seller_final:.3f}")
        print(f"Buyer combined score: {buyer_final:.3f}")
        
        logger.info(f"\nPERFORMANCE ANALYSIS:")
        logger.info(f"  • Seller score: {seller_final:.3f}")
        logger.info(f"    - Financial component: {0.4 * min(1, seller_profit / max(1, buyer_max - seller_min)):.3f}")
        logger.info(f"    - Negotiation rounds: {len(seller_agent.history)}")
        logger.info(f"    - Speed bonus: {0.2 * speed_score:.3f}")
        logger.info(f"  • Buyer score: {buyer_final:.3f}")
        logger.info(f"    - Financial component: {0.4 * min(1, buyer_savings / max(1, buyer_max - seller_min)):.3f}")
        logger.info(f"    - Negotiation rounds: {len(buyer_agent.history)}")
        logger.info(f"    - Speed bonus: {0.2 * speed_score:.3f}")

    # Final log summary
    logger.info(f"\n" + "="*60)
    logger.info(f"SESSION COMPLETED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total rounds played: {round_no}")
    logger.info(f"Log file: {log_filename}")
    logger.info("="*60)

    # Show final summary in visualization
    visualizer.show_final_summary(deal_price, deal_round, seller_profit, buyer_savings, 
                                seller_final, buyer_final)
    
    print(f"\n📁 Detailed negotiation log saved as: {log_filename}")
    print("You can review the complete negotiation process in the log file.")
    visualizer = NegotiationVisualizer(market_price, seller_min, buyer_max)
    
    seller_agent = MLAgent(role='seller', models=models, feature_list=features, persona_hint=seller_persona)
    buyer_agent  = MLAgent(role='buyer', models=models, feature_list=features, persona_hint=buyer_persona)

    seller_agent.set_context(market_price, qty, grade_code, own_limit=seller_min, opp_limit=buyer_max)
    buyer_agent.set_context(market_price, qty, grade_code, own_limit=buyer_max, opp_limit=seller_min)

    print(f"\nStarting negotiation for: {sel['name']}")
    print(f"Market reference: ₹{market_price:,}")
    print(f"Seller minimum: ₹{seller_min:,}, Buyer maximum: ₹{buyer_max:,}")
    print("Visualization window should now be open. Check your screen!\n")
    time.sleep(2)

    round_no = 0
    max_rounds = 18
    init_seller_offer = int(market_price * {"aggressive":1.25,"balanced":1.12,"diplomatic":1.08}.get(seller_persona,1.12))
    init_buyer_offer  = int(market_price * {"aggressive":0.78,"balanced":0.90,"diplomatic":0.96}.get(buyer_persona,0.90))
    init_seller_offer = max(init_seller_offer, seller_min)
    init_buyer_offer  = min(init_buyer_offer, buyer_max)

    last_seller_offer = init_seller_offer
    last_buyer_offer = init_buyer_offer

    turn = 0
    deal_price = None
    deal_round = None

    while round_no < max_rounds:
        time_left = max_rounds - round_no
        if turn == 0:
            proposal = seller_agent.propose(round_no, time_left, last_seller_offer, last_buyer_offer)
            print(f"Round {round_no} - Seller proposes: ₹{proposal:,}")
            
            # Update visualization
            visualizer.update_visualization(round_no, 'seller', proposal)
            
            accept_prob = buyer_agent.evaluate_acceptance(round_no, time_left, last_buyer_offer, proposal)
            
            # Update visualization with acceptance probability
            visualizer.accept_probs.append({
                'round': round_no,
                'prob': accept_prob,
                'role': 'buyer'  # buyer is evaluating seller's offer
            })
            visualizer.redraw_plots()
            
            if proposal <= buyer_agent.own_limit or accept_prob > 0.6:
                print(f"Buyer ACCEPTS at ₹{proposal:,} (accept_prob={accept_prob:.2f})")
                deal_price = proposal
                deal_round = round_no
                # Mark as accepted in visualization
                visualizer.negotiation_history[-1]['accepted'] = True
                visualizer.redraw_plots()
                break
            else:
                print(f"Buyer rejects (accept_prob={accept_prob:.2f})")
            last_seller_offer = proposal
        else:
            proposal = buyer_agent.propose(round_no, time_left, last_buyer_offer, last_seller_offer)
            print(f"Round {round_no} - Buyer proposes: ₹{proposal:,}")
            
            # Update visualization
            visualizer.update_visualization(round_no, 'buyer', proposal)
            
            accept_prob = seller_agent.evaluate_acceptance(round_no, time_left, last_seller_offer, proposal)
            
            # Update visualization with acceptance probability
            visualizer.accept_probs.append({
                'round': round_no,
                'prob': accept_prob,
                'role': 'seller'  # seller is evaluating buyer's offer
            })
            visualizer.redraw_plots()
            
            if proposal >= seller_agent.own_limit or accept_prob > 0.6:
                print(f"Seller ACCEPTS at ₹{proposal:,} (accept_prob={accept_prob:.2f})")
                deal_price = proposal
                deal_round = round_no
                # Mark as accepted in visualization
                visualizer.negotiation_history[-1]['accepted'] = True
                visualizer.redraw_plots()
                break
            else:
                print(f"Seller rejects (accept_prob={accept_prob:.2f})")
            last_buyer_offer = proposal

        turn = 1 - turn
        round_no += 1
        time.sleep(1)  # Slower pace for better visualization

    # Calculate final scores and display results
    if deal_price is None:
        print("\nNo deal reached within maximum rounds.")
        seller_profit = 0
        buyer_savings = 0
        seller_final = 0.0
        buyer_final = 0.0
    else:
        seller_profit = deal_price - seller_min
        buyer_savings = buyer_max - deal_price
        print(f"\n=== DEAL COMPLETED ===")
        print(f"Final price: ₹{deal_price:,} (Round {deal_round})")
        print(f"Seller profit over minimum: ₹{seller_profit:,}")
        print(f"Buyer savings under maximum: ₹{buyer_savings:,}")

        seller_char = 0.5 + 0.02 * len(seller_agent.history)
        buyer_char  = 0.5 + 0.02 * len(buyer_agent.history)
        speed_score = max(0, (max_rounds - deal_round) / max_rounds)

        seller_final = 0.4 * min(1, seller_profit / max(1, buyer_max - seller_min)) + 0.4 * min(1, seller_char) + 0.2 * speed_score
        buyer_final  = 0.4 * min(1, buyer_savings / max(1, buyer_max - seller_min)) + 0.4 * min(1, buyer_char) + 0.2 * speed_score

        print(f"\nPerformance Scores:")
        print(f"Seller combined score: {seller_final:.3f}")
        print(f"Buyer combined score: {buyer_final:.3f}")

    # Show final summary in visualization
    visualizer.show_final_summary(deal_price, deal_round, seller_profit, buyer_savings, 
                                seller_final, buyer_final)

# ---------------------------
# 6) Full flow
# ---------------------------
def main_flow():
    if not os.path.exists("negotiation_dataset.csv"):
        print("Generating synthetic dataset (this may take 20-60s)...")
        df = generate_dataset(n_negotiations=3000)
    else:
        print("Found existing negotiation_dataset.csv, loading...")
        df = pd.read_csv("negotiation_dataset.csv")

    print("Training models on dataset...")
    models = train_models(df)

    run_negotiation_session(models, models['features'])

if __name__ == "__main__":
    main_flow()
