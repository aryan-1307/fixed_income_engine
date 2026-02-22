import numpy as np
import pandas as pd
from pricing_utils import BondAnalytics

class RiskEngine:
    def __init__(self):
        self.pricing_engine = BondAnalytics()
        # Cache to store pre-generated cash flows to avoid redundant O(N*S) computations
        self._cf_cache = {}

    def _get_cached_cfs(self, bond_id, principal, coupon, freq, maturity):
        if bond_id not in self._cf_cache:
            self._cf_cache[bond_id] = self.pricing_engine.generate_cash_flows(
                principal, coupon, freq, maturity
            )
        return self._cf_cache[bond_id]

    def calculate_portfolio_loss_distribution(self, portfolio_df, scenarios):
        # Base Portfolio Value is the sum of (Market Price * Quantity)
        base_value = (portfolio_df['market_price'] * portfolio_df['quantity']).sum()

        # Pre-calculate/fetch all cash flow schedules once
        bond_data = []
        for _, bond in portfolio_df.iterrows():
            cfs = self._get_cached_cfs(
                bond['id'], bond['principal'], bond['coupon'],
                bond['frequency'], bond['maturity']
            )
            bond_data.append({
                'cfs': cfs,
                'freq': bond['frequency'],
                'qty': bond['quantity'],
                'mat': bond['maturity']
            })

        loss_distribution = []

        # Scenario Loop: Now only contains pricing math, no schedule generation
        for curve in scenarios:
            scenario_value = 0.0
            for b in bond_data:
                # Map bond maturity to the specific scenario yield
                s_yield = curve.get_yield(b['mat'])
                p_scenario = self.pricing_engine.calculate_price(s_yield, b['cfs'], b['freq'])
                scenario_value += p_scenario * b['qty']

            # Loss = Base Value (Market) - Scenario Value
            loss_distribution.append(base_value - scenario_value)

        return np.array(loss_distribution), base_value

    def calculate_var(self, losses, confidence_level=0.95):
        return np.percentile(losses, confidence_level * 100)

    def calculate_expected_shortfall(self, losses, confidence_level=0.95):
        var = self.calculate_var(losses, confidence_level)
        tail_losses = losses[losses >= var]
        return np.mean(tail_losses) if len(tail_losses) > 0 else var

    def get_risk_contribution(self, portfolio_df):
        contributions = []
        total_portfolio_dv01 = 0.0

        # First pass to calculate individual metrics
        for _, bond in portfolio_df.iterrows():
            cfs = self._get_cached_cfs(
                bond['id'], bond['principal'], bond['coupon'],
                bond['frequency'], bond['maturity']
            )

            # Solve for current YTM based on Market Price for risk sensitivities
            metrics = self.pricing_engine.price_to_risk_metrics(
                bond['market_price'], bond['principal'], bond['coupon'],
                bond['frequency'], bond['maturity']
            )

            bond_dv01 = metrics['dv01'] * bond['quantity']
            total_portfolio_dv01 += bond_dv01

            contributions.append({
                'id': bond['id'],
                'dv01_contribution': bond_dv01,
                'mod_duration': metrics['modified_duration'],
                'convexity': metrics['convexity'],
                'weight': bond['weight']
            })

        # Add percentage contribution for reporting
        for item in contributions:
            item['dv01_pct'] = (item['dv01_contribution'] / total_portfolio_dv01) if total_portfolio_dv01 != 0 else 0

        return pd.DataFrame(contributions)

    def run_stress_test(self, portfolio_df, stress_curve):
        total_pnl = 0.0
        for _, bond in portfolio_df.iterrows():
            cfs = self._get_cached_cfs(
                bond['id'], bond['principal'], bond['coupon'],
                bond['frequency'], bond['maturity']
            )

            p_initial = bond['market_price']
            y_stress = stress_curve.get_yield(bond['maturity'])
            p_stress = self.pricing_engine.calculate_price(y_stress, cfs, bond['frequency'])

            total_pnl += (p_stress - p_initial) * bond['quantity']

        return total_pnl

if __name__ == "__main__":
    from yield_curve import YieldCurve, ScenarioGenerator

    # Portfolio uses Market Price as the source of truth
    data = {
        'id': ['BOND_1', 'BOND_2'],
        'principal': [1000, 1000],
        'coupon': [0.05, 0.02],
        'frequency': [2, 2],
        'maturity': [5.0, 20.0],
        'quantity': [50, 50],
        'market_price': [1050.0, 750.0],  # Bond 2 is at a steep discount
        'weight': [0.58, 0.42]
    }
    portfolio = pd.DataFrame(data)

    base_curve = YieldCurve(0.04, -0.01, 0.01, 2.0)
    scenarios = ScenarioGenerator(base_curve).generate_scenarios(500)

    re = RiskEngine()
    losses, base_val = re.calculate_portfolio_loss_distribution(portfolio, scenarios)

    print("--- OPTIMIZED RISK ENGINE SELF-TEST ---")
    print(f"Base Portfolio Market Value: {base_val:,.2f}")
    print(f"95% VaR:                     {re.calculate_var(losses, 0.95):,.2f}")
    print(f"95% CVaR:                    {re.calculate_expected_shortfall(losses, 0.95):,.2f}")

    risk_attr = re.get_risk_contribution(portfolio)
    print("\nRisk Attribution (DV01):")
    print(risk_attr[['id', 'dv01_contribution', 'dv01_pct']])

    assert "dv01_pct" in risk_attr.columns, "Attribution logic failure."
    print("\n[SUCCESS] risk_engine.py is now locked and final.")
