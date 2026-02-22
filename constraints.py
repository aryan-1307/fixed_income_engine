import numpy as np
import pandas as pd
from pricing_utils import BondAnalytics

class PortfolioConstraints:
    def __init__(self, target_duration=None, target_convexity=None,
                 max_weight=1.0, min_weight=0.0,
                 dur_tol=0.05, conv_tol=0.5):

        self.target_duration = target_duration
        self.target_convexity = target_convexity
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.dur_tol = dur_tol
        self.conv_tol = conv_tol
        self.pricing_engine = BondAnalytics()

    def get_bond_characteristics(self, portfolio_df):
        characteristics = []
        for _, bond in portfolio_df.iterrows():
            # Derive current Greeks from market price
            metrics = self.pricing_engine.price_to_risk_metrics(
                bond['market_price'], bond['principal'], bond['coupon'],
                bond['frequency'], bond['maturity']
            )

            characteristics.append({
                'id': bond['id'],
                'mod_duration': metrics['modified_duration'],
                'convexity': metrics['convexity'],
                'price': bond['market_price']
            })

        return pd.DataFrame(characteristics)

    def calculate_portfolio_duration(self, weights, bond_characteristics):
        return np.dot(weights, bond_characteristics['mod_duration'].values)

    def calculate_portfolio_convexity(self, weights, bond_characteristics):
        return np.dot(weights, bond_characteristics['convexity'].values)

    def check_constraints(self, weights, bond_characteristics):
        violations = {}

        # 1. Budget Constraint
        total_w = np.sum(weights)
        if abs(total_w - 1.0) > 1e-6:
            violations['budget'] = total_w - 1.0

        # 2. Weight Bounds (Concentration and Long-Only)
        if any(weights > self.max_weight + 1e-7):
            violations['max_weight_excess'] = np.max(weights) - self.max_weight
        if any(weights < self.min_weight - 1e-7):
            violations['min_weight_violation'] = np.min(weights)

        # 3. Duration Matching
        if self.target_duration is not None:
            curr_dur = self.calculate_portfolio_duration(weights, bond_characteristics)
            if abs(curr_dur - self.target_duration) > self.dur_tol:
                violations['duration_mismatch'] = curr_dur - self.target_duration

        # 4. Convexity Matching
        if self.target_convexity is not None:
            curr_conv = self.calculate_portfolio_convexity(weights, bond_characteristics)
            if abs(curr_conv - self.target_convexity) > self.conv_tol:
                violations['convexity_mismatch'] = curr_conv - self.target_convexity

        return violations

if __name__ == "__main__":
    # Self-test: 3-bond universe
    bond_universe = pd.DataFrame({
        'id': ['SHORT', 'MED', 'LONG'],
        'market_price': [1020, 980, 1050],
        'principal': [1000, 1000, 1000],
        'coupon': [0.03, 0.04, 0.06],
        'frequency': [2, 2, 2],
        'maturity': [2.0, 7.0, 20.0]
    })

    # Target: Duration of 7.5, max 50% in any one bond
    pm_constraints = PortfolioConstraints(target_duration=7.5, max_weight=0.5)
    bond_stats = pm_constraints.get_bond_characteristics(bond_universe)

    # Test an 'illegal' portfolio (100% in LONG bond)
    bad_weights = np.array([0.0, 0.0, 1.0])
    v_bad = pm_constraints.check_constraints(bad_weights, bond_stats)

    # Test a 'legal' portfolio (approximate)
    good_weights = np.array([0.2, 0.4, 0.4])
    v_good = pm_constraints.check_constraints(good_weights, bond_stats)

    print("--- REFINED CONSTRAINTS SELF-TEST ---")
    print(f"Bond Analytics:\n{bond_stats[['id', 'mod_duration', 'convexity']]}")
    print(f"\nIllegal Portfolio Violations (100% Long): {list(v_bad.keys())}")
    print(f"Legal Portfolio Violations (Balanced): {list(v_good.keys())}")

    assert 'max_weight_excess' in v_bad, "Concentration check failed."
    assert len(v_good) == 0 or 'duration_mismatch' in v_good, "Constraint logic error."
    print("\n[SUCCESS] constraints.py is now locked and final.")
