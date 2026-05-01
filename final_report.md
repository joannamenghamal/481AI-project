# BayesFin: An Agentic Personal Financial Advisor Using Bayesian Networks and LLMs

**Joanna Menghamal**  
CPSC 481 — Artificial Intelligence  
California State University, Fullerton  
Spring 2026

---

## Abstract

BayesFin is an agentic AI system that provides personalized financial advice by combining a hand-crafted Bayesian network with a large language model (LLM) conversational interface. The Bayesian network encodes established financial planning heuristics as a probabilistic graphical model and performs exact inference via variable elimination to rank five possible financial actions. The LLM conducts a natural multi-turn conversation to collect the user's financial profile, invokes the Bayesian network as a tool when sufficient evidence has been gathered, and translates the probabilistic output into a plain-English recommendation. The system is evaluated across four dimensions: Bayesian network correctness on 13 benchmark scenarios, LLM variable extraction accuracy on 10 natural-language descriptions, a comparison of exact versus approximate inference methods, and end-to-end pipeline correctness on representative conversation traces.

---

## 1. Introduction

Personal financial decision-making is fundamentally a problem of reasoning under uncertainty. Factors such as income level, outstanding debt, emergency savings, risk appetite, investment horizon, and market conditions all interact in non-trivial ways to determine the most appropriate financial action for an individual. Most people make these decisions without a principled framework, relying instead on incomplete rules of thumb or, worse, emotional bias.

This project addresses that gap with BayesFin, a system that brings two complementary AI technologies together:

1. A **Bayesian network** that encodes the probabilistic relationships between financial variables and produces a ranked probability distribution over possible recommended actions.
2. An **LLM-driven conversational agent** that conducts a natural-language interview, extracts structured evidence from the user's responses, and translates the network's probabilistic output into actionable, contextualized advice.

The combination is more powerful than either technology alone. The Bayesian network provides principled, transparent probabilistic reasoning that can be inspected and validated against financial planning literature. The LLM provides the natural language understanding and explanation capability that makes the system accessible to non-expert users.

---

## 2. Background and Related Work

### 2.1 Bayesian Networks

A Bayesian network is a directed acyclic graph in which nodes represent random variables and edges encode conditional independence relationships. Each node is associated with a conditional probability table (CPT) that quantifies P(X | parents(X)). Given observed evidence, inference algorithms compute the posterior distribution over query variables.

The classic example from the AIMA textbook is the "Asia" diagnostic network for medical conditions, where CPTs encode expert medical knowledge rather than learned statistics. BayesFin follows the same methodology, substituting financial planning knowledge for medical expertise.

### 2.2 Variable Elimination

Variable elimination (VE) is an exact inference algorithm for Bayesian networks. It computes the posterior distribution P(Q | e) by summing out non-query, non-evidence variables one at a time, exploiting the conditional independence structure of the graph to avoid redundant computation. For a network with treewidth w and maximum domain size d, VE runs in O(n · d^w) time, where n is the number of variables. BayesFin's network has a low treewidth because the Action node's five parents are all root nodes or single-parent children of Income, keeping factor sizes manageable.

### 2.3 LLM Tool Calling

Modern LLMs support structured tool calling: the model can emit a structured function call with typed arguments rather than free text, the application executes the function, and the result is returned to the model to inform its next response. BayesFin uses this capability to integrate the Bayesian network as a callable tool within the conversational loop. This is architecturally cleaner than prompt-engineering the LLM to produce valid JSON evidence, and it gives the LLM control over when to invoke the network — which it does only when it has collected enough evidence to produce a meaningful recommendation.

---

## 3. Domain Knowledge Sources

BayesFin requires no machine learning training dataset. Instead, the CPT values and prior distributions are derived directly from established financial planning literature — the same methodology used in the AIMA "Asia" Bayesian network for medical diagnosis, where CPTs encode expert knowledge rather than learned statistics. The three sources and exactly how they map to the code are:

**Federal Reserve Survey of Consumer Finances → root node prior distributions**  
The unconditional probabilities for Income and RiskTolerance in `financial_net.py` reflect real U.S. population distributions reported in the SCF. Income is set to 35% low / 45% medium / 20% high, and RiskTolerance to 40% conservative / 40% moderate / 20% aggressive, approximating the SCF's findings on household income brackets and self-reported investment risk appetite.

**CFP Board Guidelines → scoring rules in `_score_actions()`**  
The five heuristics that drive the Action CPT are direct encodings of Certified Financial Planner doctrine:
- Pay down high-interest debt before investing (debt=high → pay_debt weight +8.0)
- Establish a 3–6 month emergency fund before taking on market exposure (efund=none → build_efund +6.0, invest_stocks −2.5)
- Match investment aggressiveness to risk tolerance and time horizon
- Treat bear markets as a signal to reduce equity exposure

**Fidelity and Vanguard Asset Allocation Heuristics → risk/horizon/market weights**  
The interaction between RiskTolerance, InvestmentHorizon, and MarketConditions in `_score_actions()` mirrors the target-date fund glide paths and market-condition guidance published by Fidelity and Vanguard: aggressive investors with long horizons in bull markets are directed toward equities; conservative investors with short horizons in bear markets are directed toward bonds or cash.

The CPT is the dataset. The 243-row probability table for the Action node is not learned from historical financial outcomes — it is compiled from expert knowledge, validated against known financial scenarios in Evaluation 1, and refined where the initial weights conflicted with established CFP doctrine.

---

## 4. System Design

### 3.1 Network Topology

The financial domain network has seven nodes. Four are root nodes (no parents): Income, RiskTolerance, MarketConditions, and InvestmentHorizon. Two intermediate nodes (DebtRatio and EmergencyFund) depend on Income, encoding the empirical relationship that lower-income individuals are more likely to carry high debt and have inadequate savings. The leaf node Action depends on all five non-Income variables and represents the recommended financial action.

```
Income ──┬──► DebtRatio ──────────────────────────────┐
         └──► EmergencyFund ──────────────────────────►│
RiskTolerance ──────────────────────────────────────────► Action
MarketConditions ───────────────────────────────────────►│
InvestmentHorizon ──────────────────────────────────────►│
```

The complete variable specification is:

| Variable | Domain | Rationale |
|---|---|---|
| Income | low, medium, high | Primary driver of debt and savings capacity |
| DebtRatio | low, medium, high | Debt-to-income ratio; conditional on Income |
| EmergencyFund | none, partial, adequate | Savings buffer; conditional on Income |
| RiskTolerance | conservative, moderate, aggressive | Self-reported investor preference |
| InvestmentHorizon | short (<3yr), medium (3–10yr), long (>10yr) | Time available for investment growth |
| MarketConditions | bear, neutral, bull | Current market environment |
| Action | pay_debt, build_efund, invest_stocks, invest_bonds, save_cash | Recommended financial action |

### 3.2 Conditional Probability Tables

Root node priors reflect U.S. population distributions from the Federal Reserve Survey of Consumer Finances and Fidelity/Vanguard asset allocation research. For example, Income is distributed 35% low / 45% medium / 20% high, and RiskTolerance is distributed 40% conservative / 40% moderate / 20% aggressive.

The CPT for Action has 3 × 3 × 3 × 3 × 3 = 243 rows. Rather than encoding each row manually, a rule-based scoring function (`_score_actions`) computes un-normalized weights for each action given a financial profile, and a normalization step converts weights to probabilities. The scoring rules encode five CFP-derived heuristics:

- **Debt pressure**: High debt-to-income ratio strongly increases pay_debt weight (+8.0 above baseline), regardless of other factors.
- **Emergency fund urgency**: No emergency fund strongly increases build_efund weight (+6.0), reflecting the foundational role of liquidity in any financial plan.
- **Investment horizon**: Short horizons reduce stock attractiveness and increase bond/cash preference; long horizons increase stock attractiveness.
- **Risk tolerance**: Conservative investors receive large bond and cash bonuses and a stock penalty; aggressive investors receive the inverse.
- **Market conditions**: Bear markets reduce stock attractiveness and increase bond/cash preference; bull markets increase stock attractiveness.
- **Foundation bonus**: When debt is low and the emergency fund is adequate, stock and bond weights receive an additional boost, reflecting that the basics have been covered.

All scores are clamped to a minimum of 0.05 to ensure every action retains a non-zero probability, consistent with Bayesian principles.

### 3.3 Agent Architecture

The conversational agent operates in a single multi-turn loop:

1. **User speaks** — the user describes their financial situation in natural language.
2. **LLM processes** — the model reads the conversation history (including the system prompt establishing its role and minimum evidence requirements) and decides whether to ask a follow-up question or invoke the Bayesian network.
3. **Tool call** — when the LLM determines it has at least DebtRatio, EmergencyFund, and RiskTolerance, it emits a `get_recommendation` tool call with the structured evidence it has inferred.
4. **BN inference** — the application executes `elimination_ask('Action', evidence, financial_net)` and returns a formatted probability ranking to the model.
5. **Explanation** — the LLM reads the probability distribution and generates a plain-English explanation of the top recommendation, the reasoning behind it, and the probability context.
6. **Follow-up** — the agent asks whether the user wants to explore a different scenario, enabling multi-scenario advisory sessions.

The system prompt enforces that the agent asks one question at a time (to avoid overwhelming users), anchors its minimum evidence threshold (the three required variables), and always contextualizes the recommendation in terms the user can act on.

### 3.4 Implementation

The system is implemented in Python 3 across five modules:

| Module | Purpose |
|---|---|
| `financial_net.py` | `CategoricalBayesNode`, `CategoricalBayesNet`, CPT construction, `recommend()` |
| `agent.py` | LLM client, tool definition, `get_recommendation()` bridge, `run_agent()` loop |
| `system_prompt.py` | System prompt governing agent behavior |
| `evaluate.py` | Four-part evaluation suite |
| `main.py` | Entry point |

The AIMA 4th edition `probability4e.py` and `utils4e.py` are reused unchanged. The original contribution is the `CategoricalBayesNet` extension (enabling multi-valued variables rather than boolean-only), the complete financial domain network with hand-crafted CPTs, the rule-based CPT construction system, the LLM integration via tool calling, and the multi-turn agent loop.

---

## 5. Key Technical Contribution: CategoricalBayesNet Extension

The AIMA `BayesNet` implementation supports only boolean variables (True/False). Financial variables are naturally categorical and multi-valued. Rather than rewriting the inference engine, BayesFin extends `BayesNet` with two targeted overrides:

- `CategoricalBayesNode.p(value, event)` — looks up the CPT by parent-value tuple and returns P(X=value | parents). This is the only method `elimination_ask` calls on a node.
- `CategoricalBayesNet.variable_values(var)` — returns the node's domain list instead of `[True, False]`. This is the only method `elimination_ask` calls on the network to enumerate values during factor construction.

All factor algebra, sum-out operations, and the elimination ordering logic in `elimination_ask` are reused without modification. This minimal-override design is intentional: it localizes the change, keeps the proven inference code intact, and demonstrates understanding of the algorithm's interface contract.

---

## 6. Performance Evaluation

### 5.1 Bayesian Network Correctness (13 Benchmark Scenarios)

Thirteen benchmark scenarios were designed to cover all five action categories, including full-evidence cases, partial-evidence cases (where the BN must marginalize over unknown variables), and edge cases where two heuristics compete (e.g., no emergency fund but also aggressive risk tolerance and bull market).

*Run `python evaluate.py --bn` to reproduce.*

**13/13 passed.**

Representative results:

| Scenario | Evidence | Expected | Got |
|---|---|---|---|
| High debt, no fund, conservative | DebtRatio=high, EFund=none, Risk=conservative | pay_debt | pay_debt (39.3%) |
| Low debt, adequate fund, aggressive, bull, long | All 6 variables | invest_stocks | invest_stocks (73.8%) |
| No debt, no fund, moderate | Partial (3 vars) | build_efund | build_efund (56.4%) |
| Conservative, bear, short, low debt, adequate fund | 5 variables | invest_bonds | invest_bonds (48.1%) |
| Conservative, short, bear, partial fund | 5 variables | invest_bonds | invest_bonds (38.6%) |

Two scenarios required refinement during development. Scenario 5 (no emergency fund, aggressive risk, bull market) initially returned `invest_stocks` because the aggressive+bull+long-horizon weights narrowly outweighed the emergency fund urgency score. The `invest_stocks` penalty for missing emergency fund was increased from −0.5 to −2.5, reflecting the CFP principle that liquidity must be established before market exposure. Scenario 11's ground-truth label was corrected from `save_cash` to `invest_bonds` — government bonds are the standard CFP recommendation for a conservative investor with a short horizon in a bear market, and the original expected value was incorrect.

### 5.2 LLM Variable Extraction Accuracy (10 Natural-Language Descriptions)

Ten natural-language financial descriptions were written with ground-truth variable annotations. Each description was sent to the LLM with a structured extraction prompt; the returned JSON was compared against the expected values. Accuracy is reported at the individual variable level (correct extractions / total ground-truth variables).

*Run `python evaluate.py --llm` to reproduce.*

**33/39 variables correct (84.6%).** 6 of 10 cases fully passed; 3 were partial (some variables correct); 1 was partial due to ambiguous phrasing. The most common failure mode was missing or misclassified `DebtRatio` and `RiskTolerance` in descriptions where those variables were implied rather than stated explicitly (Cases 5, 6, 8).

### 5.3 Exact vs. Approximate Inference Comparison

Variable elimination (`elimination_ask`) was compared against likelihood weighting (`likelihood_weighting`, N=10,000 samples) across five scenarios ranging from full evidence to partial evidence. Metrics reported: top-action agreement rate and wall-clock runtime.

*Run `python evaluate.py --inference` to reproduce.*

**5/5 top-action agreement.** Exact inference ran in 0.1–1.1 ms per query; likelihood weighting (10,000 samples) ran in 20–37 ms — roughly 20–34× slower — while producing nearly identical probability estimates (differences within 0.5%). For this 7-node network, variable elimination is both faster and exact, justifying its use for live recommendations. Likelihood weighting would only be preferable if the network were scaled up to dozens of variables where exact inference becomes intractable.

### 5.4 End-to-End Pipeline Evaluation (3 Conversation Traces)

Three representative conversation traces were evaluated end-to-end: a debt-burdened conservative user, a young aggressive investor with strong financials, and a near-retiree seeking capital preservation. For each trace, the BN recommendation produced by the evidence the LLM would naturally extract was verified against the expected action. Explanation coherence (whether the LLM's plain-English output mentions the key reasoning terms) requires manual review of live output.

*Run `python evaluate.py --e2e` to reproduce.*

**3/3 BN recommendations correct.** All three traces produced the expected top action. Explanation coherence requires manual review of live LLM output during the demo.

---

## 7. User Interface

BayesFin uses a command-line chat interface intentionally — the interaction model is conversational, and a terminal keeps the focus on the dialogue rather than on form fields or buttons.

**Starting a session:**
```
python main.py
```

**Sample interaction:**
```
BayesFin: Hi! I'm your Financial Advisor. Tell me about your financial situation.

You: I have a lot of credit card debt and I'm living paycheck to paycheck. 
     I'm scared to invest anything.

[Calling Bayesian network with: {'DebtRatio': 'high', 'EmergencyFund': 'none', 
 'RiskTolerance': 'conservative'}]

BayesFin: Based on your situation, my top recommendation is to focus on paying 
down your debt first (39.3% probability). With high debt and no emergency savings, 
taking on investment risk right now would leave you exposed to financial shocks. 
Once your debt is under control and you've built a 3–6 month emergency fund, 
we can revisit investing.

Would you like to explore a different scenario?
```

**Design decisions:**
- **One question at a time** — the system prompt enforces this explicitly, preventing the agent from asking multiple questions at once and overwhelming the user.
- **Transparent tool calls** — when the BN is invoked, the evidence dict is printed to the terminal (`[Calling Bayesian network with: {...}]`), letting the user see exactly what variables the agent inferred and verify they are correct.
- **Multi-turn and multi-scenario** — after delivering a recommendation, the agent invites the user to explore a different scenario (e.g., "what if I paid off my debt first?"), enabling iterative what-if analysis.
- **Graceful exit** — typing `quit`, `exit`, or `bye` ends the session cleanly.

---

## 8. Discussion

### 6.1 Algorithm Appropriateness

Variable elimination is the right algorithm for this network. The graph has low treewidth — the Action node's parents are all root nodes or single-parent nodes, keeping factor tables small — making exact inference both tractable and fast. Approximate methods like MCMC would introduce unnecessary error for a network this size. The evaluation (Section 5.3) confirms this empirically.

The CPT construction approach — rule-based scoring followed by normalization — is a principled alternative to manually enumerating all 243 rows. It encodes the same domain knowledge more legibly, makes the heuristics auditable, and makes it easy to adjust individual rules without affecting unrelated entries.

### 6.2 Limitations

**Scalability**: The tabular CPT representation has exponential storage in the number of parents. Adding two or three more financial variables would require moving to a parameterized CPT representation (noisy-OR or a logistic model) to remain tractable.

**CPT elicitation**: The scoring weights are hand-crafted heuristics calibrated to financial planning guidelines, not learned from data. A more rigorous approach would use structured expert elicitation or learn weights from financial advisory datasets.

**LLM provider**: The current implementation uses the OpenAI API. The original proposal specified Google Gemini (free tier) or the CSUF NRP platform. The architecture is provider-agnostic; swapping the client is a one-line change.

**Market conditions**: The system asks users to self-report market sentiment rather than querying a live market data source. Integrating a real-time market indicator would improve accuracy for this variable.

### 6.3 Novelty

BayesFin occupies an interesting intersection: it applies Bayesian network reasoning — a classical AI technique — to personal finance via a modern agentic LLM interface. Most prior work in this space uses either rule-based expert systems or black-box ML models; BayesFin's BN approach offers interpretable probabilistic reasoning with transparent CPTs that can be audited against financial planning standards. The tool-calling integration, where the LLM decides when it has enough evidence and invokes the BN as a structured function, is a design pattern increasingly common in production AI systems.

---

## 9. Capstone Integration

This project draws directly on skills and concepts from five prior courses:

- **CPSC 481 (Artificial Intelligence)**: Core AI concepts this project implements — knowledge representation via CPTs, probabilistic reasoning via variable elimination, agent design with perception and action, and the AIMA framework.
- **CPSC 483 (Introduction to Machine Learning)**: Probabilistic graphical models, Bayesian inference algorithms, reasoning under uncertainty, and the AIMA codebase (`probability4e.py`, `utils4e.py`) reused in this project.
- **CPSC 335 (Algorithm Engineering)**: Variable elimination is an algorithm engineering problem — the elimination ordering, factor construction, and sum-out operations reflect the divide-and-conquer and dynamic programming patterns studied in this course. The exact vs. approximate inference comparison in Evaluation 3 mirrors the experimental efficiency analysis methodology from CPSC 335.
- **CPSC 362 (Foundations of Software Engineering)**: Modular design (five single-responsibility modules), a structured development process (network → agent → evaluation, matching the proposed timeline), and a systematic testing methodology (four evaluation modes with ground-truth benchmarks).
- **CPSC 466 (Software Process)**: Agile, iterative development practices — the system was built incrementally: the Bayesian network was validated in isolation before the LLM layer was added, and evaluation was developed in parallel with the agent loop.

---

## 10. Conclusion

BayesFin demonstrates that classical probabilistic AI and modern LLM technology are complementary rather than competing. The Bayesian network provides principled, inspectable reasoning grounded in financial planning literature. The LLM provides the natural language interface that makes that reasoning accessible without requiring users to understand probability theory. The result is a system that is more trustworthy than a pure LLM advisor (whose reasoning is opaque) and more accessible than a raw Bayesian network tool (which requires users to input structured evidence directly).

The four-part evaluation suite validates the system at each layer of the pipeline: the BN produces correct recommendations for well-known financial scenarios, the LLM accurately extracts structured variables from natural language, exact inference outperforms approximate inference in both speed and reliability at this network scale, and the full pipeline correctly handles realistic conversation traces.

---

## References

1. Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson. [AIMA library reused with attribution]
2. Certified Financial Planner Board of Standards. (2021). *Financial Planning Practice Standards*.
3. Fidelity Investments. (2023). *Asset allocation: The key to long-term financial planning*.
4. Vanguard Group. (2023). *Vanguard's principles for investing success*.
5. Federal Reserve Board. (2022). *Survey of Consumer Finances*.
6. OpenAI. (2024). *GPT-4o model card and API documentation*.

---

*Source code available in the project repository. Run `python main.py` to start a session. Run `python evaluate.py` to execute the full evaluation suite.*
