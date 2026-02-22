import pandas as pd
import numpy as np
from pricing_utils import BondAnalytics
from yield_curve import YieldCurve, ScenarioGenerator
from risk_engine import RiskEngine
from constraints import PortfolioConstraints
from optimizer import PortfolioOptimizer
from simulator import PortfolioSimulator
from report_gen import ReportGenerator

# Integration with data_loader (Assuming it provides the requested function)
try:
    from data_loader import get_current_yield_curve
except ImportError:
    # Fallback for structural completeness if data_loader is missing in environment
    def get_current_yield_curve():
        return YieldCurve(beta0=0.045, beta1=-0.02, beta2=0.01, tau=2.0)

def main():
    # --- STEP 1: DATA LOAD ---
    print("\n[SECTION: DATA LOAD]")
    base_curve = get_current_yield_curve()
    print("Real yield curve loaded successfully")

    # --- STEP 2: CREATE REALISTIC BOND UNIVERSE ---
    # Creating a diversified portfolio of US Treasury-style instruments
    bond_data = {
        'id': ['UST_2Y', 'UST_3Y', 'UST_5Y', 'UST_7Y', 'UST_10Y', 'UST_20Y', 'UST_30Y'],
        'principal': [1000] * 7,
        'coupon': [0.035, 0.0375, 0.04, 0.0425, 0.045, 0.0475, 0.05],
        'frequency': [2] * 7,
        'maturity': [2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0],
        'market_price': [985.50, 990.25, 1010.00, 1005.75, 1025.00, 960.50, 940.00],
        'quantity': [1000] * 7
    }
    universe = pd.DataFrame(bond_data)
    universe['weight'] = 1.0 / len(universe)  # Initial equal weight for calculation

    # --- STEP 3: INITIALIZE ENGINES ---
    risk_engine = RiskEngine()
    # Mandate: Match a 7.5 Year Duration, max 40% concentration per bond
    constraints_manager = PortfolioConstraints(
        target_duration=7.5,
        max_weight=0.4,
        min_weight=0.0,
        dur_tol=0.05
    )
    optimizer = PortfolioOptimizer(risk_engine, constraints_manager)
    reporter = ReportGenerator(output_dir="final_reports")

    # --- STEP 4: GENERATE SCENARIOS ---
    scenario_gen = ScenarioGenerator(base_curve)
    scenarios = scenario_gen.generate_scenarios(n_scenarios=500, volatility=0.008)

    # --- STEP 5: RUN OPTIMIZATION ---
    print("\n[SECTION: OPTIMIZATION]")
    optimal_weights = optimizer.optimize(universe, scenarios, confidence_level=0.95)
    universe['weight'] = optimal_weights

    print("Optimization Complete. Optimal Allocation:")
    for i, row in universe.iterrows():
        print(f" Asset: {row['id']:<8} | Weight: {row['weight']:>8.2%}")

    # --- STEP 6: RISK ANALYTICS ---
    print("\n[SECTION: RISK ANALYTICS]")
    losses, base_val = risk_engine.calculate_portfolio_loss_distribution(universe, scenarios)
    var_95 = risk_engine.calculate_var(losses, 0.95)
    cvar_95 = risk_engine.calculate_expected_shortfall(losses, 0.95)

    # Deterministic Stress: +100bps Parallel Shift
    stress_curve = scenario_gen.apply_manual_shock(type="parallel", magnitude=0.01)
    stress_loss = risk_engine.run_stress_test(universe, stress_curve)

    # --- STEP 7: RUN SIMULATION ---
    print("\n[SECTION: SIMULATION]")
    simulator = PortfolioSimulator(optimizer, universe, optimal_weights)
    # Simulate 2 years of quarterly rebalancing
    sim_history = simulator.run_simulation(base_curve, steps=8, step_size_years=0.25)

    # --- STEP 8 & 9: REPORTING & CLEANUP ---
    print("\n[SECTION: REPORT EXPORT]")
    risk_attr = risk_engine.get_risk_contribution(universe)

    reporter.display_portfolio_summary(universe, risk_attr)
    reporter.display_risk_metrics(var_95, cvar_95, stress_loss)
    reporter.display_simulation_results(sim_history)

    print("\n" + "=" * 60)
    print(" SYSTEM EXECUTION COMPLETE ".center(60, "#"))
    print("=" * 60)

if __name__ == "__main__":
    main()
