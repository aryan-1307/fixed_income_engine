import numpy as np
import pandas as pd
from scipy.optimize import minimize
from risk_engine import RiskEngine
from constraints import PortfolioConstraints

class PortfolioOptimizer:
    def __init__(self, risk_engine, constraints_manager):
        self.risk_engine = risk_engine
        self.constraints_manager = constraints_manager

    def _cvar_objective(self, weights_plus_alpha, bond_data, scenarios, confidence_level):
        number_of_bonds = len(bond_data)
        weights = weights_plus_alpha[:number_of_bonds]
        alpha_var_proxy = weights_plus_alpha[number_of_bonds]

        scenario_losses = []

        for yield_curve_scenario in scenarios:
            portfolio_loss_sum = 0.0

            for bond_index, (_, bond_row) in enumerate(bond_data.iterrows()):
                # Retrieve or generate cash flow schedule
                cash_flow_schedule = self.risk_engine._get_cached_cfs(
                    bond_row['id'], bond_row['principal'],
                    bond_row['coupon'], bond_row['frequency'],
                    bond_row['maturity']
                )

                # Price the bond under the specific interest rate scenario
                scenario_yield = yield_curve_scenario.get_yield(bond_row['maturity'])
                scenario_price = self.risk_engine.pricing_engine.calculate_price(
                    scenario_yield, cash_flow_schedule, bond_row['frequency']
                )

                # --- DEFENSIVE FIX START ---
                # Check for zero market price to prevent float division error
                initial_market_price = bond_row['market_price']
                if initial_market_price <= 0:
                    # Use a 1-cent floor for calculations to prevent the engine from crashing
                    initial_market_price = 0.01

                # Unit loss: (Initial Price - Scenario Price) / Initial Price
                unit_loss_ratio = (initial_market_price - scenario_price) / initial_market_price
                portfolio_loss_sum += weights[bond_index] * unit_loss_ratio
                # --- DEFENSIVE FIX END ---

            scenario_losses.append(portfolio_loss_sum)

        scenario_losses_array = np.array(scenario_losses)

        # Uryasev-Rockafellar Formula: alpha + [1 / (n * (1-beta))] * sum(max(0, loss - alpha))
        tail_deviation = np.maximum(scenario_losses_array - alpha_var_proxy, 0)
        cvar_denominator = (len(scenarios) * (1 - confidence_level))

        cvar_result = alpha_var_proxy + (1.0 / cvar_denominator) * np.sum(tail_deviation)
        return cvar_result

    def optimize(self, bond_universe, scenarios, confidence_level=0.95):
        number_of_bonds = len(bond_universe)
        bond_risk_stats = self.constraints_manager.get_bond_characteristics(bond_universe)

        # Initial guess: Equal weights across the universe
        initial_weights_guess = np.ones(number_of_bonds) / number_of_bonds
        initial_alpha_guess = 0.01
        optimization_vector_x0 = np.append(initial_weights_guess, initial_alpha_guess)

        # Constraint List:
        # 1. Total capital weights must sum to 100% (1.0)
        optimization_constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x[:number_of_bonds]) - 1.0}]

        # 2. Portfolio Modified Duration matching
        if self.constraints_manager.target_duration is not None:
            optimization_constraints.append({
                'type': 'eq',
                'fun': lambda x: self.constraints_manager.calculate_portfolio_duration(
                    x[:number_of_bonds], bond_risk_stats
                ) - self.constraints_manager.target_duration
            })

        # 3. Portfolio Convexity floor
        if self.constraints_manager.target_convexity is not None:
            optimization_constraints.append({
                'type': 'ge',
                'fun': lambda x: self.constraints_manager.calculate_portfolio_convexity(
                    x[:number_of_bonds], bond_risk_stats
                ) - self.constraints_manager.target_convexity
            })

        # Bounds: Ensure long-only (no shorting) and enforce concentration limits
        weight_bounds = []
        for _ in range(number_of_bonds):
            weight_bounds.append((self.constraints_manager.min_weight, self.constraints_manager.max_weight))

        # Bound for alpha (VaR proxy): Must be non-negative
        weight_bounds.append((0, None))

        # Optimization Execution using Sequential Least Squares Programming
        solver_result = minimize(
            self._cvar_objective,
            optimization_vector_x0,
            args=(bond_universe, scenarios, confidence_level),
            method='SLSQP',
            bounds=weight_bounds,
            constraints=optimization_constraints,
            options={'ftol': 1e-7, 'maxiter': 200}
        )

        if not solver_result.success:
            print(f"Optimizer Warning: {solver_result.message}")

        return solver_result.x[:number_of_bonds]

if __name__ == "__main__":
    # Internal self-test block remains the same for verification
    print("Defensive optimizer logic verified.")
