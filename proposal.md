# BayesFin: An Agentic Personal Financial Advisor Using Bayesian Networks and LLMs

**Joanna Menghamal**  
CPSC 481 — Artificial Intelligence  
California State University, Fullerton

---

## Problem Description

Making sound personal financial decisions requires reasoning under uncertainty across many interrelated factors: income stability, existing debt, savings adequacy, risk appetite, investment timeline, and market conditions. Most people make these decisions without a principled framework, often relying on incomplete rules of thumb or emotional bias.

This project builds **BayesFin**, an agentic AI financial advisor that combines two complementary technologies:

1. A **Bayesian network** that encodes probabilistic relationships between financial variables and produces a ranked distribution over possible recommended actions (e.g., pay down debt, build an emergency fund, invest in equities or bonds).
2. An **LLM-driven conversational agent** that interviews the user in plain English, extracts structured financial variables from their responses, identifies what information is still missing, asks targeted follow-up questions, and then translates the Bayesian network's output into a clear, actionable explanation.

The system is **multi-turn**: the agent maintains dialogue state, iteratively collects evidence, and updates its recommendations as new information is provided.

---

## Programming Language

Python 3

---

## Datasets

No machine learning training dataset is required. The Bayesian network's **Conditional Probability Tables (CPTs)** are derived from established financial planning frameworks:

- Certified Financial Planner (CFP) Board guidelines
- Fidelity and Vanguard asset allocation heuristics
- Federal Reserve Survey of Consumer Finances (as domain knowledge, not training data)

This follows the same methodology as the classic AIMA "Asia" Bayesian network for medical diagnosis, where CPTs encode expert knowledge rather than learned statistics.

---

## Existing Code and Extensions

**Reused from prior coursework (CPSC 483):**

- `probability4e.py` — Full Bayesian network implementation from the AIMA 4th edition codebase, including `BayesNet`, `BayesNode`, exact inference via `elimination_ask()`, and approximate inference via `likelihood_weighting()` and `gibbs_ask()`
- `utils4e.py` — Supporting utility functions (sampling, distributions, data structures)

**Original contributions:**

- A new financial domain Bayesian network with custom topology and CPTs (the primary AI contribution)
- Integration with a free LLM API (Google Gemini free tier or CSUF's NRP platform) for natural language understanding and explanation generation
- A multi-turn agent loop that tracks known vs. unknown financial variables and drives the conversation until sufficient evidence is collected for meaningful inference
- A command-line chat interface

---

## Algorithm and Approach

The system operates in four stages:

**Stage 1 — Variable Extraction (LLM)**  
The LLM receives the user's free-text description of their financial situation. A structured prompt instructs it to extract six financial variables: Income Level, Debt-to-Income Ratio, Emergency Fund Status, Risk Tolerance, Investment Horizon, and Market Sentiment. The LLM returns a JSON object with known values and flags for unknowns.

**Stage 2 — Clarification Loop (Agent)**  
For each unknown variable, the agent generates a natural-language follow-up question. This loop continues until enough evidence is present for meaningful Bayesian inference (at minimum: income level, debt ratio, and risk tolerance).

**Stage 3 — Bayesian Inference (BN)**  
The collected evidence is passed to `elimination_ask()` from the AIMA library, which computes the posterior probability distribution:

> P(RecommendedAction | Income, DebtRatio, EmergencyFund, RiskTolerance, Horizon, Market)

The network returns a probability ranking over five possible actions:
`pay_debt`, `build_emergency_fund`, `invest_stocks`, `invest_bonds`, `save_cash`

**Stage 4 — Explanation (LLM)**  
The posterior is passed back to the LLM, which generates a plain-English explanation of the top recommendation, the reasoning behind it, and the probability context.

**Bayesian Network Structure:**

| Node | Values |
|---|---|
| Income Level | low, medium, high |
| Debt-to-Income Ratio | low, medium, high |
| Emergency Fund | none, partial, adequate |
| Risk Tolerance | conservative, moderate, aggressive |
| Investment Horizon | short (<3yr), medium (3–10yr), long (>10yr) |
| Market Conditions | bear, neutral, bull |
| Recommended Action | pay_debt, build_emergency_fund, invest_stocks, invest_bonds, save_cash |

---

## Timeline

| Dates | Task |
|---|---|
| Apr 7–13 | Finalize BN topology and CPTs from financial literature |
| Apr 14–20 | Implement `financial_net.py`; test inference with hard-coded evidence |
| Apr 21–27 | Integrate LLM API; build variable extraction prompt |
| Apr 28–May 1 | Build multi-turn agent loop; connect full pipeline |
| May 2–4 | Testing: benchmark scenarios + LLM extraction accuracy |
| May 5–7 | In-class presentation |
| May 8–15 | Final report + code submission |

---

## Special Computing Platform

None. The application runs on a standard laptop. LLM access is via Google Gemini (free tier) or the CSUF NRP platform (free for enrolled students), avoiding any paid API dependency.

---

## Roles and Responsibilities

Solo project. All components — network design, CPT elicitation, Bayesian inference implementation, LLM integration, agent loop, evaluation, presentation, and written report — are completed by Joanna Menghamal.

---

## Capstone Integration

This project draws on skills from multiple prior courses:

- **CPSC 481 (Artificial Intelligence)**: Core AI concepts this project directly implements — knowledge representation, probabilistic reasoning, search, and agent design; the AIMA framework used for Bayesian inference originates here
- **CPSC 483 (Introduction to Machine Learning)**: Bayesian inference algorithms, probabilistic graphical models, supervised/unsupervised learning, and reasoning under uncertainty; provided the AIMA library used in this project
- **CPSC 335 (Algorithm Engineering)**: Algorithm design and asymptotic analysis applied to the variable elimination algorithm; efficiency comparison between exact and approximate inference methods in Evaluation 3
- **CPSC 362 (Foundations of Software Engineering)**: Software engineering principles — modular design, structured development process, and testing methodology applied throughout the agent and evaluation pipeline
- **CPSC 466 (Software Process)**: Agile development practices and iterative process improvement; informed the incremental build approach (network → agent loop → LLM integration → evaluation)

---

## Performance Evaluation

1. **BN correctness**: 10–15 benchmark financial scenarios with known ground-truth recommendations (e.g., "high debt + no emergency fund → pay debt / build fund"). The network's top-ranked action is compared against the expected advice.
2. **LLM extraction accuracy**: 10 natural-language financial descriptions are fed to the agent; extracted variables are checked for correctness.
3. **Inference comparison**: Variable elimination (exact) vs. likelihood weighting (approximate) — measured on output agreement and runtime across varying levels of evidence.
4. **End-to-end evaluation**: Full conversation traces (user input → agent questions → BN inference → explanation) are rated for recommendation correctness and explanation coherence.
