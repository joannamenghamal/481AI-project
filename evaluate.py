"""
evaluate.py  —  BayesFin evaluation suite

Four evaluation modes matching the proposal:

  1. BN correctness     – 13 benchmark scenarios with known ground-truth
  2. LLM extraction     – 10 natural-language descriptions; checks extracted vars
  3. Inference comparison – elimination_ask vs likelihood_weighting agreement/runtime
  4. End-to-end traces  – full conversation snippets rated for correctness

Run individual sections:
  python evaluate.py --bn
  python evaluate.py --llm
  python evaluate.py --inference
  python evaluate.py --e2e
  python evaluate.py          (runs all four)
"""

import argparse
import time
import os
import json
from financial_net import recommend, ACTION_DOMAIN, financial_net
from probability4e import likelihood_weighting


# ===========================================================================
# 1. BN CORRECTNESS  (13 benchmark scenarios)
# ===========================================================================

BENCHMARKS = [
    # --- debt-priority cases ------------------------------------------------
    {
        "label": "High debt, no emergency fund, conservative",
        "evidence": {"DebtRatio": "high", "EmergencyFund": "none", "RiskTolerance": "conservative"},
        "expected": "pay_debt",
    },
    {
        "label": "High debt, no emergency fund, aggressive",
        "evidence": {"DebtRatio": "high", "EmergencyFund": "none", "RiskTolerance": "aggressive"},
        "expected": "pay_debt",
    },
    {
        "label": "High debt, adequate fund, moderate, bull market",
        "evidence": {"DebtRatio": "high", "EmergencyFund": "adequate", "RiskTolerance": "moderate",
                     "MarketConditions": "bull", "InvestmentHorizon": "long"},
        "expected": "pay_debt",
    },
    # --- emergency-fund cases -----------------------------------------------
    {
        "label": "No debt, no emergency fund, any risk",
        "evidence": {"DebtRatio": "low", "EmergencyFund": "none", "RiskTolerance": "moderate"},
        "expected": "build_efund",
    },
    {
        "label": "No debt, no emergency fund, aggressive, bull market",
        "evidence": {"DebtRatio": "low", "EmergencyFund": "none", "RiskTolerance": "aggressive",
                     "MarketConditions": "bull", "InvestmentHorizon": "long"},
        "expected": "build_efund",
    },
    # --- stock investment cases ---------------------------------------------
    {
        "label": "Low debt, adequate fund, aggressive, bull market, long horizon",
        "evidence": {"DebtRatio": "low", "EmergencyFund": "adequate", "RiskTolerance": "aggressive",
                     "MarketConditions": "bull", "InvestmentHorizon": "long"},
        "expected": "invest_stocks",
    },
    {
        "label": "High income, low debt, adequate fund, moderate, neutral, medium horizon",
        "evidence": {"Income": "high", "DebtRatio": "low", "EmergencyFund": "adequate",
                     "RiskTolerance": "moderate", "MarketConditions": "neutral",
                     "InvestmentHorizon": "medium"},
        "expected": "invest_stocks",
    },
    {
        "label": "Low debt, adequate fund, aggressive, neutral market, medium horizon",
        "evidence": {"DebtRatio": "low", "EmergencyFund": "adequate", "RiskTolerance": "aggressive",
                     "MarketConditions": "neutral", "InvestmentHorizon": "medium"},
        "expected": "invest_stocks",
    },
    # --- bond / conservative cases -----------------------------------------
    {
        "label": "Conservative, bear market, short horizon, low debt, adequate fund",
        "evidence": {"DebtRatio": "low", "EmergencyFund": "adequate", "RiskTolerance": "conservative",
                     "MarketConditions": "bear", "InvestmentHorizon": "short"},
        "expected": "invest_bonds",
    },
    {
        "label": "Conservative, adequate fund, low debt, neutral market, long horizon",
        "evidence": {"DebtRatio": "low", "EmergencyFund": "adequate", "RiskTolerance": "conservative",
                     "MarketConditions": "neutral", "InvestmentHorizon": "long"},
        "expected": "invest_bonds",
    },
    # --- save-cash cases ----------------------------------------------------
    {
        "label": "Conservative, short horizon, bear market, partial fund",
        "evidence": {"DebtRatio": "low", "EmergencyFund": "partial", "RiskTolerance": "conservative",
                     "MarketConditions": "bear", "InvestmentHorizon": "short"},
        "expected": "invest_bonds",  # conservative + bear + short horizon → bonds over cash
    },
    # --- partial-evidence cases (agent may not have all vars) ---------------
    {
        "label": "Partial evidence: only debt and risk known (high debt, conservative)",
        "evidence": {"DebtRatio": "high", "RiskTolerance": "conservative"},
        "expected": "pay_debt",
    },
    {
        "label": "Partial evidence: no fund, moderate risk — build fund expected",
        "evidence": {"EmergencyFund": "none", "RiskTolerance": "moderate"},
        "expected": "build_efund",
    },
]


def run_bn_evaluation():
    print("=" * 60)
    print("EVALUATION 1: BN Correctness")
    print("=" * 60)
    passed = 0

    for i, scenario in enumerate(BENCHMARKS, 1):
        dist = recommend(scenario["evidence"])
        ranked = sorted(ACTION_DOMAIN, key=lambda a: dist[a], reverse=True)
        top = ranked[0]
        expected = scenario["expected"]
        ok = top == expected
        if ok:
            passed += 1

        label = "PASS" if ok else "FAIL"
        print(f"[{label}] Scenario {i}: {scenario['label']}")
        print(f"    Expected: {expected}")
        print(f"    Got:      {top} ({dist[top]:.1%})")
        if not ok:
            print(f"    Full ranking: {[(a, f'{dist[a]:.1%}') for a in ranked]}")
        print()

    print(f"Result: {passed}/{len(BENCHMARKS)} passed\n")
    return passed, len(BENCHMARKS)


# ===========================================================================
# 2. LLM EXTRACTION ACCURACY
#    Requires OPENAI_API_KEY in environment.
#    Each description is sent to the LLM; extracted JSON is checked against
#    the ground-truth variable values.
# ===========================================================================

LLM_EXTRACTION_CASES = [
    {
        "description": "I make about $60,000 a year, have a car loan and some credit card debt "
                       "that's pretty high relative to my income. I don't have any savings set aside "
                       "for emergencies. I'm pretty nervous about losing money.",
        "ground_truth": {
            "Income": "medium",
            "DebtRatio": "high",
            "EmergencyFund": "none",
            "RiskTolerance": "conservative",
        },
    },
    {
        "description": "I'm a software engineer earning six figures. My only debt is a small "
                       "student loan. I have six months of expenses saved up and I'm comfortable "
                       "taking on market risk for long-term gains.",
        "ground_truth": {
            "Income": "high",
            "DebtRatio": "low",
            "EmergencyFund": "adequate",
            "RiskTolerance": "aggressive",
        },
    },
    {
        "description": "I work part-time and barely cover my bills each month. I have a lot of "
                       "credit card debt and nothing saved for an emergency. I'm terrified of "
                       "investing because I can't afford to lose anything.",
        "ground_truth": {
            "Income": "low",
            "DebtRatio": "high",
            "EmergencyFund": "none",
            "RiskTolerance": "conservative",
        },
    },
    {
        "description": "I have a 401(k) and a small brokerage account. My mortgage is my main "
                       "debt but it's manageable. I have about two months of expenses in savings — "
                       "not quite where I want it. I'd say I'm somewhere in the middle on risk.",
        "ground_truth": {
            "DebtRatio": "medium",
            "EmergencyFund": "partial",
            "RiskTolerance": "moderate",
        },
    },
    {
        "description": "I just graduated and started my first job. I don't have any debt yet "
                       "but I also haven't saved anything. I want to invest but I'm not sure where "
                       "to start. I'm thinking long term, at least 10 years.",
        "ground_truth": {
            "DebtRatio": "low",
            "EmergencyFund": "none",
            "RiskTolerance": "moderate",
            "InvestmentHorizon": "long",
        },
    },
    {
        "description": "I'm retiring in about two years. I've paid off my house and have a "
                       "good pension coming. My savings are solid. I just want to protect what "
                       "I have — I don't need growth.",
        "ground_truth": {
            "DebtRatio": "low",
            "EmergencyFund": "adequate",
            "RiskTolerance": "conservative",
            "InvestmentHorizon": "short",
        },
    },
    {
        "description": "Stocks are tanking right now and everyone's worried. I have medium income, "
                       "some debt from my car, and I'm thinking about whether to invest or wait.",
        "ground_truth": {
            "Income": "medium",
            "DebtRatio": "medium",
            "MarketConditions": "bear",
        },
    },
    {
        "description": "I earn about $45K a year. I have a little bit of credit card debt but "
                       "nothing crazy. I keep about one month of expenses in my checking account. "
                       "I'd rather not risk my money but I'm open to some growth.",
        "ground_truth": {
            "Income": "medium",
            "DebtRatio": "low",
            "EmergencyFund": "partial",
            "RiskTolerance": "moderate",
        },
    },
    {
        "description": "The market is doing really well right now. I have no debt, plenty in "
                       "savings, and I make a high salary. I'm in my 30s so I have decades to "
                       "invest. I'm willing to take on a lot of risk.",
        "ground_truth": {
            "Income": "high",
            "DebtRatio": "low",
            "EmergencyFund": "adequate",
            "RiskTolerance": "aggressive",
            "MarketConditions": "bull",
            "InvestmentHorizon": "long",
        },
    },
    {
        "description": "I owe a lot on my credit cards, maybe 40% of my income. I have almost "
                       "nothing saved. I don't really know much about investing so I prefer "
                       "safe options.",
        "ground_truth": {
            "DebtRatio": "high",
            "EmergencyFund": "none",
            "RiskTolerance": "conservative",
        },
    },
]

EXTRACTION_SYSTEM_PROMPT = """You are a financial data extractor.
Given a user's description of their financial situation, extract known variables and return ONLY valid JSON.

Variables and their ONLY valid values:
- Income: "low", "medium", "high"
- DebtRatio: "low", "medium", "high"
- EmergencyFund: "none", "partial", "adequate"
- RiskTolerance: "conservative", "moderate", "aggressive"
- MarketConditions: "bear", "neutral", "bull"
- InvestmentHorizon: "short", "medium", "long"

Rules:
- Return ONLY a JSON object with any subset of the 6 variable names as keys
- Omit variables you cannot determine (do not include them)
- Use ONLY the exact string values listed above
"""


def run_llm_extraction_evaluation():
    print("=" * 60)
    print("EVALUATION 2: LLM Variable Extraction Accuracy")
    print("=" * 60)

    try:
        from openai import OpenAI
        from dotenv import load_dotenv
        load_dotenv()
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    except Exception as e:
        print(f"Skipped: could not initialize LLM client ({e})\n")
        return None, None

    total_vars = 0
    correct_vars = 0

    for i, case in enumerate(LLM_EXTRACTION_CASES, 1):
        try:
            response = client.responses.create(
                model="gpt-4o-mini",
                input=[
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user",   "content": case["description"]},
                ],
                text={"format": {"type": "json_object"}},
            )
            extracted = json.loads(response.output_text)
        except Exception as e:
            print(f"[ERROR] Case {i}: {e}\n")
            continue

        gt = case["ground_truth"]
        case_correct = sum(1 for k, v in gt.items() if extracted.get(k) == v)
        case_total   = len(gt)
        total_vars  += case_total
        correct_vars += case_correct

        status = "PASS" if case_correct == case_total else "PARTIAL" if case_correct > 0 else "FAIL"
        print(f"[{status}] Case {i} ({case_correct}/{case_total} vars correct)")
        for k, v in gt.items():
            got = extracted.get(k, "<missing>")
            mark = "✓" if got == v else "✗"
            print(f"    {mark} {k}: expected={v}, got={got}")
        print()

    if total_vars:
        pct = correct_vars / total_vars * 100
        print(f"Result: {correct_vars}/{total_vars} variables correct ({pct:.1f}%)\n")
    return correct_vars, total_vars


# ===========================================================================
# 3. INFERENCE COMPARISON: exact (elimination_ask) vs approximate (likelihood_weighting)
# ===========================================================================

INFERENCE_SCENARIOS = [
    {"Income": "high", "DebtRatio": "low", "EmergencyFund": "adequate",
     "RiskTolerance": "aggressive", "MarketConditions": "bull", "InvestmentHorizon": "long"},
    {"DebtRatio": "high", "EmergencyFund": "none", "RiskTolerance": "conservative"},
    {"DebtRatio": "low", "EmergencyFund": "none", "RiskTolerance": "moderate"},
    {"Income": "medium", "DebtRatio": "medium", "EmergencyFund": "partial",
     "RiskTolerance": "moderate", "MarketConditions": "neutral"},
    {"DebtRatio": "low", "EmergencyFund": "adequate", "RiskTolerance": "conservative",
     "MarketConditions": "bear", "InvestmentHorizon": "short"},
]

N_SAMPLES = 10_000   # for likelihood weighting


def run_inference_comparison():
    print("=" * 60)
    print("EVALUATION 3: Exact vs Approximate Inference")
    print(f"(likelihood weighting uses {N_SAMPLES:,} samples)")
    print("=" * 60)

    agreements = 0

    for i, evidence in enumerate(INFERENCE_SCENARIOS, 1):
        # --- Exact ---
        t0 = time.perf_counter()
        exact_dist = recommend(evidence)
        exact_time = time.perf_counter() - t0
        exact_top  = max(ACTION_DOMAIN, key=lambda a: exact_dist[a])

        # --- Approximate ---
        t0 = time.perf_counter()
        approx_dist = likelihood_weighting('Action', evidence, financial_net, N_SAMPLES)
        approx_time = time.perf_counter() - t0
        approx_top  = max(ACTION_DOMAIN, key=lambda a: approx_dist[a])

        agree = exact_top == approx_top
        if agree:
            agreements += 1

        print(f"Scenario {i}: {evidence}")
        print(f"  Exact      → {exact_top} ({exact_dist[exact_top]:.1%})  [{exact_time*1000:.1f} ms]")
        print(f"  Approx     → {approx_top} ({approx_dist[approx_top]:.1%})  [{approx_time*1000:.1f} ms]")
        print(f"  Agreement  : {'YES' if agree else 'NO'}")

        # Show full ranking side-by-side
        print(f"  {'Action':<22} {'Exact':>8} {'Approx':>8}")
        for a in sorted(ACTION_DOMAIN, key=lambda x: exact_dist[x], reverse=True):
            print(f"  {a:<22} {exact_dist[a]:>7.1%} {approx_dist[a]:>7.1%}")
        print()

    print(f"Top-action agreement: {agreements}/{len(INFERENCE_SCENARIOS)} scenarios\n")
    return agreements, len(INFERENCE_SCENARIOS)


# ===========================================================================
# 4. END-TO-END EVALUATION
#    Simulated conversation traces: user input → expected BN call args →
#    expected top recommendation.  Checks that the pipeline produces the
#    right recommendation given a realistic conversation.
# ===========================================================================

E2E_TRACES = [
    {
        "label": "Debt-burdened user, conservative",
        "conversation": [
            ("user",      "I have a lot of credit card debt and nothing saved up. I hate risk."),
            ("assistant", "Got it — sounds like your debt-to-income ratio is high, no emergency fund, "
                          "and you're risk-averse. Let me check what the network says."),
        ],
        "bn_evidence":    {"DebtRatio": "high", "EmergencyFund": "none", "RiskTolerance": "conservative"},
        "expected_action": "pay_debt",
        "expected_explanation_keywords": ["debt", "pay", "interest"],
    },
    {
        "label": "Young aggressive investor, strong financials",
        "conversation": [
            ("user",      "I make great money, no debt, solid emergency fund. I want to grow my wealth "
                          "aggressively over the next 15 years. Market looks good."),
            ("assistant", "Perfect situation for long-term equity investment. Let me run the numbers."),
        ],
        "bn_evidence":    {"Income": "high", "DebtRatio": "low", "EmergencyFund": "adequate",
                           "RiskTolerance": "aggressive", "MarketConditions": "bull",
                           "InvestmentHorizon": "long"},
        "expected_action": "invest_stocks",
        "expected_explanation_keywords": ["stock", "invest", "long"],
    },
    {
        "label": "Near-retiree, conservative, short horizon",
        "conversation": [
            ("user",      "I'm retiring in 18 months. I've paid everything off, good savings. "
                          "I just want to protect my nest egg."),
            ("assistant", "Short horizon + conservative profile. Let me see the recommendation."),
        ],
        "bn_evidence":    {"DebtRatio": "low", "EmergencyFund": "adequate",
                           "RiskTolerance": "conservative", "InvestmentHorizon": "short",
                           "MarketConditions": "bear"},
        "expected_action": "invest_bonds",
        "expected_explanation_keywords": ["bond", "safe", "protect"],
    },
]


def run_e2e_evaluation():
    print("=" * 60)
    print("EVALUATION 4: End-to-End Pipeline")
    print("=" * 60)

    passed = 0

    for i, trace in enumerate(E2E_TRACES, 1):
        dist    = recommend(trace["bn_evidence"])
        ranked  = sorted(ACTION_DOMAIN, key=lambda a: dist[a], reverse=True)
        top     = ranked[0]
        ok      = top == trace["expected_action"]
        if ok:
            passed += 1

        print(f"Trace {i}: {trace['label']}")
        print(f"  Conversation excerpt:")
        for role, text in trace["conversation"]:
            print(f"    [{role}] {text[:80]}{'...' if len(text) > 80 else ''}")
        print(f"  BN evidence:       {trace['bn_evidence']}")
        print(f"  Expected action:   {trace['expected_action']}")
        print(f"  BN top action:     {top} ({dist[top]:.1%})  [{'PASS' if ok else 'FAIL'}]")
        print(f"  Explanation should mention: {trace['expected_explanation_keywords']}")
        print(f"  (Explanation coherence requires manual review of live LLM output)")
        print()

    print(f"BN recommendation correctness: {passed}/{len(E2E_TRACES)} traces\n")
    return passed, len(E2E_TRACES)


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="BayesFin evaluation suite")
    parser.add_argument("--bn",        action="store_true", help="BN correctness benchmarks")
    parser.add_argument("--llm",       action="store_true", help="LLM extraction accuracy")
    parser.add_argument("--inference", action="store_true", help="Exact vs approximate inference")
    parser.add_argument("--e2e",       action="store_true", help="End-to-end traces")
    args = parser.parse_args()

    run_all = not any([args.bn, args.llm, args.inference, args.e2e])

    if run_all or args.bn:
        run_bn_evaluation()
    if run_all or args.llm:
        run_llm_extraction_evaluation()
    if run_all or args.inference:
        run_inference_comparison()
    if run_all or args.e2e:
        run_e2e_evaluation()


if __name__ == "__main__":
    main()
