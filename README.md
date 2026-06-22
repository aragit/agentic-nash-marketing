# 🏛️ Nash Marketing Agents

**Multi-Agent Competitive Ad Auction with Nash Equilibrium**

&gt; *"We need builders who ship. Not prototypes. Not demos."*

---

## What This Is

Autonomous AI brand agents compete in real-time ad auctions. Each agent uses an LLM to formulate bidding strategy, then a game-theoretic Nash equilibrium solver computes optimal mixed strategies. Budget guardrails prevent catastrophic depletion.

**Built entirely with AI as the primary development interface** — every file was designed, coded, and iterated using Claude Code and Cursor.

---

## 🎯 Why This Matters for the Role

| Job Requirement | How This Project Delivers |
|:---|:---|
| **Ship end-to-end** | Idea → Architecture → Implementation → Docker Deployment → Dashboard |
| **Production software** | FastAPI + PostgreSQL/SQLite + pytest (36 tests) + Docker + real API |
| **Full-stack builder** | Backend (Python) + API (FastAPI) + Database (SQLAlchemy) + Frontend (vanilla JS + Chart.js) + Deployment (Docker) |
| **AI-first development** | LLM is the PRIMARY bidding strategist; documented AI workflow |
| **Autonomous agents** | Self-correcting multi-agent system with continuous simulation |
| **AI reviews AI** | Guardrails audit agent decisions; Nash solver validates strategies |

---

## 🏗️ Architecture

1.An agentic AI system is defined by autonomous entities that perceive, decide, and act in an environment with persistent goals. Our system satisfies all four criteria:

```Table
Criterion	Implementation	Evidence
Perception	Agents observe market state (clearing price, competitor count, win rate)	BrandAgent.decide_bid(market_price, competitor_count, ...)
Decision	LLM-powered strategic reasoning with structured JSON output	LLMEngine.chat_completion() generates bid strategy
Action	Agents submit bids to auction engine	AuctionEngine.run_round() executes bids
Persistent goals	Budget preservation, CPA targets, win rate optimization	AgentState tracks cumulative performance
```
Unlike simple API wrappers, these agents maintain state across rounds, adapt strategy based on outcomes, and operate without human intervention — the definition of autonomy.

2. Architecture Paradigm: Neuro-Symbolic with Game-Theoretic Orchestration

```text

┌─────────────────────────────────────────────────────────────────────────────┐
│                    NEURO-SYMBOLIC ARCHITECTURE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────┐        ┌─────────────────────┐                     │
│  │   NEURAL (LLM)      │        │   SYMBOLIC (Math)   │                     │
│  │                     │        │                     │                     │
│  │  • Pattern matching │◄──────►│  • Nash equilibrium │                     │
│  │  • Strategy gen     │        │  • Linear programs  │                     │
│  │  • Natural lang     │        │  • Constraint opt   │                     │
│  │  • Context aware    │        │  • Probabilistic    │                     │
│  │                     │        │    inference        │                     │
│  └─────────────────────┘        └─────────────────────┘                     │
│           │                              │                                   │
│           └──────────────┬───────────────┘                                   │
│                          │                                                  │
│                          ▼                                                  │
│               ┌─────────────────────┐                                       │
│               │  HYBRID REASONING   │                                       │
│               │  Layer              │                                       │
│               │                     │                                       │
│               │  LLM proposes →     │                                       │
│               │  Symbolic validates → │                                       │
│               │  Guardrail enforces   │                                       │
│               └─────────────────────┘                                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```
*The LLM is the "intuition" — it generates creative bidding strategies. The Nash solver is the "logic" — it proves no agent can improve by deviating. The guardrail is the "conscience" — it prevents self-destructive behavior.*

3. The Nash Algorithm: Why Game Theory Matters
#### The Problem: The Tragedy of the Commons in Ad Auctions
**Without equilibrium analysis, agents engage in destructive bidding wars:**

Scenario: 3 brands, 100 impressions/day, each with $5000 budget

Without Nash:
  Nike bids $10 → Adidas bids $11 → Puma bids $12 → Nike bids $13...
  Result: Everyone's CPA explodes. Budgets deplete in days.

With Nash Equilibrium:
  Solver computes: "If Nike bids $3.20, Adidas bids $2.80, Puma bids $2.50,
  no brand can improve by changing unilaterally."
  Result: Stable market. Sustainable budgets. Predictable CPAs.  

**How It Works: Iterative Best-Response with Softmax**

```python

# Pseudocode of our Nash solver
for iteration in range(max_iterations):
    for each_agent in agents:
        # Compute expected utility for every bid level
        # given opponents' current mixed strategies
        utilities = [expected_profit(bid, opponent_strategies) 
                     for bid in bid_levels]
        
        # Softmax best response (temperature annealing)
        # High temp early = exploration. Low temp late = convergence.
        new_strategy = softmax(utilities / temperature)
    
    # Check convergence: did anyone's strategy change significantly?
    if max_strategy_change < tolerance:
        break  # Nash equilibrium found!
```

***Mathematical guarantee:*** At convergence, no agent can improve their expected utility by changing their strategy alone. This is the ***definition of Nash equilibrium.***



---

## 🚀 Quick Start

### Docker (Recommended)

```bash
git clone https://github.com/aragit/nash-marketing-agents.git
cd nash-marketing-agents
docker-compose up --build

```

Open http://localhost:8000 for the dashboard.

**Local Development**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn api.main:app --reload
```
📊 Dashboard Features
- System Health: Real-time API, LLM backend, and database status
- Quick Stats: Total simulations, last clearing price, total revenue
- Run Simulation: Configure agents, rounds, and strategies
- Budget Depletion Chart: Visualize spending vs. remaining per agent
- Win Rate Chart: Doughnut chart of auction success
- Clearing Price History: Line chart of market dynamics
- Nash Equilibrium: Expected bid distribution per agent
- Event Log: Real-time simulation events

## 🧪 Testing
```bash
pytest tests/ -v
```

```text
36 tests covering:
- Agent behavior (initialization, bidding, state updates, CPA calculation)
- Auction mechanics (empty auction, revenue matching, budget depletion)
- Nash equilibrium (convergence, strategy validity, clearing price)
- Budget guardrails (soft warning, hard cap, emergency mode)
- API endpoints (health, simulation, Nash compute)
```


## 🎮 How It Works
Configure: Set brand names, strategies (aggressive/balanced/conservative), budgets, and target CPAs
Simulate: Each round, agents use LLM reasoning to decide bids
Auction: Second-price VCG mechanism allocates impressions
Equilibrium: Post-hoc Nash solver computes optimal mixed strategies
Analyze: Dashboard visualizes budget curves, win rates, and price history

## 🔮 Future Integration
This project is designed to integrate with https://github.com/aragit/autonomous-procurement-swarm:
```Table
Procurement Swarm	Nash Marketing Agents	Integration
Bilateral negotiation	N-player competitive auction	Shared LLM engine
Buyer vs. Seller	Brand vs. Brand	Shared PostgreSQL ledger
Pareto efficiency	Nash equilibrium	Unified dashboard
Cost minimization	Budget preservation	Cross-domain analytics
```
## 📦 Tech Stack
```Table
Layer	Technology
LLM	MockLLM (default) / Transformers CPU (optional)
Math	NumPy + SciPy (Nash equilibrium, optimization)
Database	SQLite (local) / PostgreSQL (production)
API	FastAPI + Pydantic v2
ORM	SQLAlchemy 2.0
Dashboard	Vanilla JS + Chart.js
Container	Docker + docker-compose
Testing	pytest + pytest-asyncio
```

## 📝 API Endpoints
Table
Method	Endpoint	Description
GET	/health	System health check
POST	/simulation/run	Run new auction simulation
GET	/simulations	List all simulations
GET	/simulation/{id}	Get simulation details
POST	/nash/compute	Compute Nash equilibrium


**📄 License**
MIT
