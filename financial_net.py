"""
financial_net.py

Extends the AIMA BayesNet/BayesNode to support categorical (multi-valued)
variables, then defines the BayesFin financial decision network.

Why extend instead of rewrite?
  elimination_ask() from probability4e.py only calls three things on the
  network/nodes:
    - bn.variable_values(var)   → we override this to return our domain list
    - node.p(value, event)      → we implement this for categorical CPTs
    - node.parents              → plain attribute, no change needed
  Everything else (the factor algebra, sum-out, etc.) is reused unchanged.
"""

import random
from itertools import product
from probability4e import BayesNet, elimination_ask, ProbDist


# ---------------------------------------------------------------------------
# 1. CategoricalBayesNode
#    A node whose variable can take any finite set of string values,
#    not just True/False.
# ---------------------------------------------------------------------------

class CategoricalBayesNode:
    """
    Conditional probability distribution for a categorical variable X.

    CPT format:
      { (parent_val_1, parent_val_2, ...): {x_val: prob, ...}, ... }

    For root nodes (no parents) use an empty tuple key:
      { (): {'low': 0.3, 'medium': 0.5, 'high': 0.2} }

    Probabilities for each parent combination must sum to 1.0.
    """

    def __init__(self, variable, parents, domain, cpt):
        """
        variable : str   — variable name
        parents  : str or list — space-separated parent names, or list
        domain   : list  — ordered list of possible values, e.g. ['low','medium','high']
        cpt      : dict  — maps parent-value tuples to {value: probability} dicts
        """
        if isinstance(parents, str):
            parents = parents.split()
        self.variable = variable
        self.parents  = parents
        self.domain   = domain
        self.cpt      = cpt
        self.children = []      # filled in by CategoricalBayesNet.add()

        # Validate that every row sums to 1 (within floating-point tolerance)
        for parent_vals, dist in cpt.items():
            assert abs(sum(dist.values()) - 1.0) < 1e-6, (
                f"CPT row {parent_vals} for '{variable}' sums to {sum(dist.values()):.4f}, not 1.0"
            )
            assert set(dist.keys()) == set(domain), (
                f"CPT row {parent_vals} for '{variable}' has wrong keys: {set(dist.keys())}"
            )

    def sample(self, event):
        """
        Sample a value from P(X | parent values in event).
        Required by likelihood_weighting in probability4e.py.
        """
        parent_vals = tuple(event[par] for par in self.parents)
        dist = self.cpt[parent_vals]
        r = random.random()
        cumulative = 0.0
        for value, prob in dist.items():
            cumulative += prob
            if r < cumulative:
                return value
        return list(dist.keys())[-1]  # fallback for floating-point edge case

    def p(self, value, event):
        """
        Return P(X=value | parent values drawn from event).
        This is the method elimination_ask relies on.
        """
        parent_vals = tuple(event[par] for par in self.parents)
        return self.cpt[parent_vals][value]

    def __repr__(self):
        return f"CategoricalBayesNode({self.variable!r}, parents={self.parents})"


# ---------------------------------------------------------------------------
# 2. CategoricalBayesNet
#    Subclass of BayesNet that knows how to add CategoricalBayesNode objects
#    and returns the correct domain for each variable.
# ---------------------------------------------------------------------------

class CategoricalBayesNet(BayesNet):
    """
    A Bayesian network whose variables may take categorical (string) values.
    Inherits the full AIMA BayesNet infrastructure; only overrides:
      - add()             to accept CategoricalBayesNode specs
      - variable_values() to return the node's domain instead of [True, False]
    """

    def add(self, node_spec):
        """
        node_spec: (variable, parents, domain, cpt)
        Parents must already be in the network (topological order).
        """
        variable, parents, domain, cpt = node_spec
        node = CategoricalBayesNode(variable, parents, domain, cpt)
        assert node.variable not in self.variables, \
            f"Variable '{node.variable}' already in network"
        assert all(p in self.variables for p in node.parents), \
            f"All parents of '{node.variable}' must be added first"
        self.nodes.append(node)
        self.variables.append(node.variable)
        for parent in node.parents:
            self.variable_node(parent).children.append(node)

    def variable_values(self, var):
        """Return the domain (list of possible values) for variable var."""
        return self.variable_node(var).domain


# ---------------------------------------------------------------------------
# 3. Financial domain knowledge helpers
#
#    We build the CPT for the Action node programmatically using financial
#    planning heuristics.  This is cleaner than manually writing 243 rows.
#
#    Key rules encoded (from CFP guidelines):
#      - High debt → strongly prefer pay_debt regardless of other factors
#      - No emergency fund → strongly prefer build_efund (except if high debt)
#      - Conservative risk + short horizon → prefer bonds/cash
#      - Aggressive risk + long horizon + bull market → prefer stocks
#      - Bear market softens stock preference for moderate/aggressive investors
# ---------------------------------------------------------------------------

INCOME_DOMAIN     = ('low', 'medium', 'high')
DEBT_DOMAIN       = ('low', 'medium', 'high')
EFUND_DOMAIN      = ('none', 'partial', 'adequate')
RISK_DOMAIN       = ('conservative', 'moderate', 'aggressive')
HORIZON_DOMAIN    = ('short', 'medium', 'long')
MARKET_DOMAIN     = ('bear', 'neutral', 'bull')
ACTION_DOMAIN     = ('pay_debt', 'build_efund', 'invest_stocks', 'invest_bonds', 'save_cash')


def _score_actions(debt, efund, risk, market, horizon):
    """
    Return a raw score dict {action: weight} for a given financial profile.
    Weights are un-normalised; they will be converted to probabilities.

    Think of each weight as 'how strongly does this situation call for
    this action?'.  Higher = more likely recommendation.
    """
    scores = {a: 1.0 for a in ACTION_DOMAIN}   # baseline: all actions plausible

    # ---- Rule 1: Debt pressure ------------------------------------------
    if debt == 'high':
        scores['pay_debt']        += 8.0
        scores['invest_stocks']   -= 0.8
        scores['invest_bonds']    -= 0.5
    elif debt == 'medium':
        scores['pay_debt']        += 3.0
    # debt == 'low' → no adjustment

    # ---- Rule 2: Emergency fund urgency ---------------------------------
    if efund == 'none':
        scores['build_efund']     += 6.0
        scores['invest_stocks']   -= 2.5   # CFP: fund emergency fund before investing
    elif efund == 'partial':
        scores['build_efund']     += 2.0

    # ---- Rule 3: Investment horizon -------------------------------------
    if horizon == 'short':
        scores['invest_stocks']   -= 1.5
        scores['save_cash']       += 2.0
        scores['invest_bonds']    += 1.0
    elif horizon == 'long':
        scores['invest_stocks']   += 2.0
        scores['save_cash']       -= 0.5

    # ---- Rule 4: Risk tolerance -----------------------------------------
    if risk == 'conservative':
        scores['invest_stocks']   -= 2.0
        scores['invest_bonds']    += 3.0
        scores['save_cash']       += 1.0
    elif risk == 'aggressive':
        scores['invest_stocks']   += 3.0
        scores['invest_bonds']    -= 0.5
        scores['save_cash']       -= 0.5

    # ---- Rule 5: Market conditions affect stock attractiveness ----------
    if market == 'bear':
        scores['invest_stocks']   -= 2.0
        scores['save_cash']       += 1.5
        scores['invest_bonds']    += 1.0
    elif market == 'bull':
        scores['invest_stocks']   += 2.0


    # When basics are covered, shift weight toward investing
    if debt == 'low' and efund == 'adequate':
        scores['invest_stocks'] += 2.0
        scores['invest_bonds']  += 1.0

    # Clamp scores to a small positive minimum so no action has zero probability
    scores = {a: max(s, 0.05) for a, s in scores.items()}
    return scores


def _normalise(scores):
    """Convert raw score dict to a probability dict summing to 1."""
    total = sum(scores.values())
    return {a: s / total for a, s in scores.items()}


def _build_action_cpt():
    """
    Build the full CPT for the Action node by iterating over all
    (debt, efund, risk, market, horizon) combinations.
    Returns a dict: {(debt, efund, risk, market, horizon): {action: prob}}
    """
    cpt = {}
    for debt, efund, risk, market, horizon in product(
            DEBT_DOMAIN, EFUND_DOMAIN, RISK_DOMAIN, MARKET_DOMAIN, HORIZON_DOMAIN):
        raw = _score_actions(debt, efund, risk, market, horizon)
        cpt[(debt, efund, risk, market, horizon)] = _normalise(raw)
    return cpt


# ---------------------------------------------------------------------------
# 4. Build the network
#
#    Node order must be topological (parents before children).
#
#    Network structure:
#
#      Income ──┬──► DebtRatio ──────────────────────────┐
#               └──► EmergencyFund ──────────────────────►│
#      RiskTolerance ──────────────────────────────────────► Action
#      MarketConditions ───────────────────────────────────►│
#      InvestmentHorizon ──────────────────────────────────►│
#
#    Income influences DebtRatio and EmergencyFund (higher income →
#    less likely to carry high debt or have no emergency fund).
#    Everything feeds into Action.
# ---------------------------------------------------------------------------

financial_net = CategoricalBayesNet([
    # ---- Root nodes (no parents) ----------------------------------------

    ('Income', '', INCOME_DOMAIN, {
        (): {'low': 0.35, 'medium': 0.45, 'high': 0.20}
    }),

    ('RiskTolerance', '', RISK_DOMAIN, {
        (): {'conservative': 0.40, 'moderate': 0.40, 'aggressive': 0.20}
    }),

    ('MarketConditions', '', MARKET_DOMAIN, {
        (): {'bear': 0.25, 'neutral': 0.50, 'bull': 0.25}
    }),

    ('InvestmentHorizon', '', HORIZON_DOMAIN, {
        (): {'short': 0.30, 'medium': 0.40, 'long': 0.30}
    }),

    # ---- Intermediate nodes (depend on Income) --------------------------

    # DebtRatio | Income
    # Lower income → more likely to carry high debt-to-income ratio
    ('DebtRatio', 'Income', DEBT_DOMAIN, {
        ('low',):    {'low': 0.15, 'medium': 0.35, 'high': 0.50},
        ('medium',): {'low': 0.35, 'medium': 0.45, 'high': 0.20},
        ('high',):   {'low': 0.60, 'medium': 0.30, 'high': 0.10},
    }),

    # EmergencyFund | Income
    # Lower income → less likely to have an adequate emergency fund
    ('EmergencyFund', 'Income', EFUND_DOMAIN, {
        ('low',):    {'none': 0.55, 'partial': 0.35, 'adequate': 0.10},
        ('medium',): {'none': 0.25, 'partial': 0.45, 'adequate': 0.30},
        ('high',):   {'none': 0.10, 'partial': 0.30, 'adequate': 0.60},
    }),

    # ---- Leaf node (the recommendation) --------------------------------

    # Action | DebtRatio, EmergencyFund, RiskTolerance, MarketConditions, InvestmentHorizon
    ('Action',
     'DebtRatio EmergencyFund RiskTolerance MarketConditions InvestmentHorizon',
     list(ACTION_DOMAIN),
     _build_action_cpt()),
])


# ---------------------------------------------------------------------------
# 5. Convenience function for querying the network
# ---------------------------------------------------------------------------

def recommend(evidence: dict) -> ProbDist:
    """
    Query the network for P(Action | evidence).

    evidence: dict mapping variable names to observed values, e.g.
        {
            'Income':           'medium',
            'DebtRatio':        'high',
            'EmergencyFund':    'none',
            'RiskTolerance':    'conservative',
            'MarketConditions': 'neutral',
            'InvestmentHorizon':'short',
        }

    Returns a ProbDist object.  Access probabilities with dist['pay_debt'] etc.
    Call .show_approx() for a sorted string representation.
    """
    return elimination_ask('Action', evidence, financial_net)


# ---------------------------------------------------------------------------
# 6. Quick sanity test (run this file directly to verify)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=== BayesFin Sanity Tests ===\n")

    test_cases = [
        {
            'label': "High debt, no emergency fund, conservative — expect pay_debt or build_efund",
            'evidence': {
                'Income': 'low',
                'DebtRatio': 'high',
                'EmergencyFund': 'none',
                'RiskTolerance': 'conservative',
                'MarketConditions': 'neutral',
                'InvestmentHorizon': 'short',
            }
        },
        {
            'label': "Low debt, adequate efund, aggressive, long horizon, bull market — expect invest_stocks",
            'evidence': {
                'Income': 'high',
                'DebtRatio': 'low',
                'EmergencyFund': 'adequate',
                'RiskTolerance': 'aggressive',
                'MarketConditions': 'bull',
                'InvestmentHorizon': 'long',
            }
        },
        {
            'label': "Partial efund, conservative, bear market — expect build_efund or invest_bonds",
            'evidence': {
                'Income': 'medium',
                'DebtRatio': 'low',
                'EmergencyFund': 'partial',
                'RiskTolerance': 'conservative',
                'MarketConditions': 'bear',
                'InvestmentHorizon': 'medium',
            }
        },
        {
            'label': "No debt, adequate efund — moderate investor, neutral market (partial evidence)",
            'evidence': {
                'DebtRatio': 'low',
                'EmergencyFund': 'adequate',
                'RiskTolerance': 'moderate',
                'MarketConditions': 'neutral',
            }
        },
    ]

    for tc in test_cases:
        print(f"Scenario: {tc['label']}")
        print(f"Evidence: {tc['evidence']}")
        dist = recommend(tc['evidence'])
        ranked = sorted(ACTION_DOMAIN, key=lambda a: dist[a], reverse=True)
        print("Recommendations (highest → lowest probability):")
        for action in ranked:
            print(f"  {action:<25} {dist[action]:.3f}")
        print(f"  Top recommendation: {ranked[0].upper()}\n")
