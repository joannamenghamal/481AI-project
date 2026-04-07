from financial_net import recommend, ACTION_DOMAIN

# Each scenario has:
#   evidence - inputs to the BN (can be partial)
#   expected - the action we expect to rank #1
#   label - human-readable description

BENCHMARKS = [
    {
        "label": "High debt, no emergency fun, conservative",
        "evidence": {"DebtRatio": "high", "EmergencyFund": "none", "RiskTolerance": "conservative"},
        "expected": "pay_debt",
    },
    {
        "label": "Low debt, adequate fund, aggressive, bull market, long horizon",
        "evidence": {"DebtRatio": "low", "EmergencyFund": "adequate", "RiskTolerance": "aggressive",
                     "MarketConditions": "bull", "InvestmentHorizon": "long"},
        "expected": "invest_stocks",
    },
    {
        "label": "No debt, no emergency fund, any risk",
        "evidence": {"DebtRatio": "low", "EmergencyFund": "none", "RiskTolerance": "moderate"},
        "expected": "build_efund",
    },
    {
        "label": "Conservative, bear market, short horizon",
        "evidence": {"DebtRatio": "low", "EmergencyFund": "adequate", "RiskTolerance": "conservative",
                     "MarketConditions": "bear", "InvestmentHorizon": "short"},
        "expected": "invest_bonds",
    },
    {
        "label": "High income, low debt, adequate fund, moderate, neutral, medium horizon",
        "evidence": {"Income": "high", "DebtRatio": "low", "EmergencyFund": "adequate",
                     "RiskTolerance": "moderate", "MarketConditions": "neutral", "InvestmentHorizon": "medium"},
        "expected": "invest_stocks",
    },
    
]

def run_evaluation():
    passed = 0

    for i, scenario in enumerate(BENCHMARKS, 1):
        dist = recommend(scenario["evidence"])
        ranked = sorted(ACTION_DOMAIN, key=lambda a: dist[a], reverse=True)
        top = ranked[0]
        expected = scenario["expected"]
        result = "PASS" if top == expected else "FAIL"
        if top == expected:
            passed += 1

        print(f"[{result}] Scenario {i}: {scenario['label']}")
        print(f"    Expected: {expected}")
        print(f"    Got: {top} ({dist[top]:.1%})")
        if result == "FAIL":
            print(f"    Full ranking: {[(a, f'{dist[a]:.1%}') for a in ranked]}")
        print()

    print(f"Result: {passed}/{len(BENCHMARKS)} passed") 


if __name__ == "__main__":
    run_evaluation()