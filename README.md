# BayesFin — Agentic Financial Advisor

BayesFin is a personal financial advisor that combines a hand-crafted Bayesian network with a GPT-powered conversational agent. You describe your financial situation in plain English; the agent extracts key variables, queries the Bayesian network for a probabilistic recommendation, and explains the result in plain English.

Built for CPSC 481 — Artificial Intelligence, California State University Fullerton.

---

## How it works

The system has two moving parts that work together:

1. **The LLM agent** (GPT-4o-mini) holds a conversation with the user. It asks questions in natural language, figures out the user's financial profile, and decides when it has enough information to ask for a recommendation.
2. **The Bayesian network** does the actual reasoning. Once the agent has collected the user's financial variables, it calls the BN as a tool and gets back a probability distribution over five possible actions.

```
User types free-form text
        │
        ▼
  GPT-4o-mini agent
  (asks follow-up questions, extracts variables)
        │
        │  when ready → tool call: get_recommendation(DebtRatio, EmergencyFund, RiskTolerance, ...)
        ▼
  Bayesian Network  →  P(Action | evidence)
        │
        ▼
  Agent translates the probability distribution into a plain-English explanation
        │
        ▼
  User sees advice + reasoning
```

The conversation is multi-turn: the agent keeps chatting and refining its picture of the user's finances before making a call.

---

## Code layout

```
481AI-project/
├── main.py              ← run this to start the chatbot
├── agent.py             ← the GPT agent + tool bridge
├── system_prompt.py     ← instructions that control the agent's behavior
├── financial_net.py     ← the Bayesian network (the AI core)
├── evaluate.py          ← test suite (4 evaluation modes)
├── probability4e.py     ← AIMA inference engine (reused from CPSC 483)
└── utils4e.py           ← AIMA utility helpers (reused from CPSC 483)
```

### `main.py`
The entry point. It just calls `run_agent()` from `agent.py`. Nothing else lives here.

### `agent.py`
This is the conversational loop. It does three things:

- **Defines the tool** (`get_recommendation`) that the LLM can call when it wants a BN query. The tool schema tells GPT exactly which variables to pass and what values are valid.
- **Runs the inner loop**: sends the conversation to GPT, checks whether GPT returned a tool call or a user-facing message, and handles both. If GPT calls the tool, the agent runs the BN and appends the result back into the conversation so GPT can read it and explain the output.
- **Bridges the two systems**: converts GPT's tool arguments (a plain dict) into a `recommend()` call, then formats the BN's probability distribution as a readable string that GPT uses to write its explanation.

### `system_prompt.py`
A single string — the system prompt loaded into the LLM at the start of every conversation. It tells the agent:
- What its role is (financial advisor, not a general chatbot)
- What minimum evidence it needs before calling the BN (`DebtRatio`, `EmergencyFund`, `RiskTolerance`)
- How to explain results after getting BN output
- To ask if the user wants to explore another scenario

Keeping this in its own file makes it easy to tune the agent's behavior without touching any logic.

### `financial_net.py`
The core AI component. This file:

1. **Defines `CategoricalBayesNode`** — an extension of the AIMA `BayesNode` that works with string-valued variables (e.g., `'low'`/`'medium'`/`'high'`) instead of just `True`/`False`. The key method is `p(value, event)`, which returns the probability of a variable taking a given value given its parents' values.

2. **Defines `CategoricalBayesNet`** — a subclass of the AIMA `BayesNet` that uses `CategoricalBayesNode` objects and overrides `variable_values()` to return each node's string domain. The rest of the AIMA inference engine (factor algebra, variable elimination) works unchanged.

3. **Encodes the financial domain** as a Bayesian network with this structure:

   ```
   Income ──┬──► DebtRatio ──────────────────────────┐
            └──► EmergencyFund ──────────────────────►│
   RiskTolerance ──────────────────────────────────────► Action
   MarketConditions ───────────────────────────────────►│
   InvestmentHorizon ──────────────────────────────────►│
   ```

   `Income` is the only node with a causal influence on other intermediate nodes — higher income makes high debt and no emergency fund less likely. Everything feeds into `Action`.

4. **Builds the Action CPT programmatically** using `_score_actions()`. Instead of manually writing all 243 rows (3 debt × 3 efund × 3 risk × 3 market × 3 horizon), it assigns a score to each action based on five weighted rules derived from CFP guidelines, then normalizes the scores into probabilities.

5. **Exposes `recommend(evidence)`** — the function the rest of the code calls. It wraps `elimination_ask()` from the AIMA library and returns a probability distribution over the five actions.

**Variables and their domains:**

| Variable | Values |
|---|---|
| Income | `low` · `medium` · `high` |
| DebtRatio | `low` · `medium` · `high` |
| EmergencyFund | `none` · `partial` · `adequate` |
| RiskTolerance | `conservative` · `moderate` · `aggressive` |
| MarketConditions | `bear` · `neutral` · `bull` |
| InvestmentHorizon | `short` · `medium` · `long` |
| **Action** | `pay_debt` · `build_efund` · `invest_stocks` · `invest_bonds` · `save_cash` |

### `evaluate.py`
A standalone test suite with four evaluation modes, each runnable independently via `--bn`, `--llm`, `--inference`, or `--e2e`. See the [Evaluation](#evaluation) section below.

### `probability4e.py` and `utils4e.py`
Reused from CPSC 483 coursework — the AIMA 4th edition Bayesian network library. Provides:
- `BayesNet` / `BayesNode` — the base classes extended in `financial_net.py`
- `elimination_ask()` — exact inference via variable elimination
- `likelihood_weighting()` — approximate inference via weighted sampling
- `ProbDist` — a probability distribution class

These files are not modified. BayesFin subclasses and extends them.

---

## Setup

**Requirements:** Python 3.10+, an OpenAI API key.

```bash
pip install openai python-dotenv
```

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
```

---

## Running

```bash
python main.py
```

Example session:

```
BayesFin: Hi! I'm your Financial Advisor. Tell me about your financial situation.

You: I have a lot of credit card debt and nothing saved up. I hate risk.

[Calling Bayesian network with: {'DebtRatio': 'high', 'EmergencyFund': 'none', 'RiskTolerance': 'conservative'}]

BayesFin: Given your high debt and no emergency fund, the network strongly recommends
focusing on paying down debt first (71.2%), followed by building your emergency fund
(18.4%). Investing should wait until these foundations are in place.
```

Type `quit` or `exit` to end the session.

---

## Evaluation

```bash
python evaluate.py             # run all four
python evaluate.py --bn        # BN correctness
python evaluate.py --llm       # LLM extraction accuracy
python evaluate.py --inference # exact vs approximate inference
python evaluate.py --e2e       # end-to-end pipeline
```

| Mode | What it tests |
|---|---|
| `--bn` | 13 hand-labeled financial scenarios; checks that the BN's top action matches the expected CFP-grounded recommendation |
| `--llm` | 10 natural-language descriptions; checks that GPT-4o-mini extracts the correct variable values from free-text input |
| `--inference` | Compares `elimination_ask` (exact) vs `likelihood_weighting` (approximate, 10k samples) on top-action agreement and wall-clock runtime |
| `--e2e` | Simulated conversation traces verified for correct BN recommendation; explanation coherence requires manual review |

---

## Acknowledgements

- Inference engine: AIMA 4th edition Python code (`probability4e.py`, `utils4e.py`) from prior coursework (CPSC 483).
- Financial heuristics: CFP Board guidelines, Fidelity/Vanguard asset allocation models.
