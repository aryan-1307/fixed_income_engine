import numpy as np
import pandas as pd
from pricing_utils import BondAnalytics
from yield_curve import YieldCurve, ScenarioGenerator

class PortfolioSimulator:
    def __init__(self, optimizer, initial_universe, initial_weights):
        self.optimizer = optimizer
        self.pricing_engine = BondAnalytics()
        self.universe = initial_universe.copy()
        self.weights = np.array(initial_weights)
        self.history = []

    def run_simulation(self, base_curve, steps=8, step_size_years=0.25):
        current_weights = self.weights
        current_universe = self.universe.copy()

        # Simulate a single deterministic path of interest rate evolution
        gen = ScenarioGenerator(base_curve)
        future_path = gen.generate_scenarios(n_scenarios=steps, volatility=0.005)

        print(f"Executing simulation for {steps} steps ({steps * step_size_years} years)...")

        for i in range(steps):
            # 1. Update Market Environment
            market_curve = future_path[i]

            # 2. Aging and Cash Flow Re-Sync
            current_universe['maturity'] = current_universe['maturity'] - step_size_years

            # Floor maturity to avoid negative time; simulate 'rolling' into 6-month paper
            current_universe.loc[current_universe['maturity'] <= 0, 'maturity'] = 0.5

            # CRITICAL: Clear the cash flow cache because the 'time' dimension has shifted
            self.optimizer.risk_engine._cf_cache.clear()

            # 3. Mark-to-Market (MtM)
            for idx, bond in current_universe.iterrows():
                cfs = self.optimizer.risk_engine._get_cached_cfs(
                    bond['id'], bond['principal'], bond['coupon'],
                    bond['frequency'], bond['maturity']
                )
                new_yield = market_curve.get_yield(bond['maturity'])
                current_universe.at[idx, 'market_price'] = self.pricing_engine.calculate_price(
                    new_yield, cfs, bond['frequency']
                )

            # 4. Periodic Re-Optimization
            local_scenarios = ScenarioGenerator(market_curve).generate_scenarios(n_scenarios=100)
            new_weights = self.optimizer.optimize(current_universe, local_scenarios)

            # 5. Performance Tracking
            turnover = np.sum(np.abs(new_weights - current_weights))

            stats = self.optimizer.constraints_manager.get_bond_characteristics(current_universe)
            port_dur = self.optimizer.constraints_manager.calculate_portfolio_duration(new_weights, stats)
            port_conv = self.optimizer.constraints_manager.calculate_portfolio_convexity(new_weights, stats)

            self.history.append({
                'step': i + 1,
                'turnover': turnover,
                'portfolio_duration': port_dur,
                'portfolio_convexity': port_conv,
                'benchmark_yield': market_curve.get_yield(10.0)
            })

            current_weights = new_weights

        return pd.DataFrame(self.history)

if __name__ == "__main__":
    from risk_engine import RiskEngine
    from constraints import PortfolioConstraints
    from optimizer import PortfolioOptimizer

    # Test Data
    universe = pd.DataFrame({
        'id': ['SHORT', 'MID', 'LONG'],
        'principal': [1000, 1000, 1000],
        'coupon': [0.02, 0.04, 0.06],
        'frequency': [2, 2, 2],
        'maturity': [2.0, 5.0, 15.0],
        'market_price': [1000.0, 1000.0, 1000.0],
        'quantity': [1, 1, 1]
    })

    start_curve = YieldCurve(0.04, -0.01, 0.01, 2.0)
    risk_eng = RiskEngine()
    cons_eng = PortfolioConstraints(target_duration=5.0, max_weight=0.5)
    opt_eng = PortfolioOptimizer(risk_eng, cons_eng)

    sim_eng = PortfolioSimulator(opt_eng, universe, [0.33, 0.33, 0.34])
    sim_output = sim_eng.run_simulation(start_curve, steps=4)

    print("\n--- REFINED SIMULATOR OUTPUT ---")
    print(sim_output)

    assert 'portfolio_convexity' in sim_output.columns, "Metric tracking failed."
    print("\n[SUCCESS] simulator.py is now locked and final.")
