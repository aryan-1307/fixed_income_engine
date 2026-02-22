import pandas as pd
import os

class ReportGenerator:
    def __init__(self, output_dir="reports"):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _print_header(self, title):
        print("\n" + "="*60)
        print(f" {title.upper()} ".center(60, "-"))
        print("="*60)

    def display_portfolio_summary(self, portfolio_df, risk_attr_df):
        self._print_header("Portfolio Composition & Risk Attribution")

        # Merge attribution data for a single view
        summary = pd.merge(
            portfolio_df[['id', 'market_price', 'weight']],
            risk_attr_df[['id', 'mod_duration', 'dv01_pct']],
            on='id'
        )

        print(f"{'Bond ID':<12} | {'Price':>10} | {'Weight':>8} | {'Duration':>10} | {'DV01 %':>8}")
        print("-" * 60)

        for _, row in summary.iterrows():
            print(f"{row['id']:<12} | {row['market_price']:>10.2f} | {row['weight']:>8.2%} | "
                  f"{row['mod_duration']:>10.4f} | {row['dv01_pct']:>8.2%}")

        # Export to CSV
        summary.to_csv(os.path.join(self.output_dir, "portfolio_summary.csv"), index=False)
        print(f"\n[INFO] Summary exported to {self.output_dir}/portfolio_summary.csv")

    def display_risk_metrics(self, var, cvar, stress_loss):
        self._print_header("Risk Analytics (Scenario Based)")

        print(f"{'Metric':<25} | {'Value':>20}")
        print("-" * 50)
        print(f"{'95% Value at Risk (VaR)':<25} | {var:>20.2f}")
        print(f"{'95% Expected Shortfall':<25} | {cvar:>20.2f}")
        print(f"{'Deterministic Stress PnL':<25} | {stress_loss:>20.2f}")

        # Log metrics
        with open(os.path.join(self.output_dir, "risk_metrics.txt"), "w") as f:
            f.write(f"VaR_95: {var}\nCVaR_95: {cvar}\nStress_Loss: {stress_loss}")

    def display_simulation_results(self, sim_history):
        self._print_header("Simulation Performance History")

        print(f"{'Step':<5} | {'Turnover':>10} | {'Portfolio Dur':>15} | {'Bench Yield':>12}")
        print("-" * 60)

        for _, row in sim_history.iterrows():
            print(f"{int(row['step']):<5} | {row['turnover']:>10.4%} | "
                  f"{row['portfolio_duration']:>15.4f} | {row['benchmark_yield']:>12.2%}")

        sim_history.to_csv(os.path.join(self.output_dir, "simulation_history.csv"), index=False)
        print(f"\n[INFO] History exported to {self.output_dir}/simulation_history.csv")

if __name__ == "__main__":
    # Self-test
    reporter = ReportGenerator(output_dir="test_reports")

    # Mock Data
    port_mock = pd.DataFrame({
        'id': ['BOND_1', 'BOND_2'],
        'market_price': [1050.0, 980.0],
        'weight': [0.6, 0.4]
    })
    attr_mock = pd.DataFrame({
        'id': ['BOND_1', 'BOND_2'],
        'mod_duration': [4.5, 12.2],
        'dv01_pct': [0.35, 0.65]
    })
    sim_mock = pd.DataFrame({
        'step': [1, 2],
        'turnover': [0.05, 0.02],
        'portfolio_duration': [7.9, 8.0],
        'benchmark_yield': [0.04, 0.042]
    })

    reporter.display_portfolio_summary(port_mock, attr_mock)
    reporter.display_risk_metrics(5000.0, 7500.0, -12000.0)
    reporter.display_simulation_results(sim_mock)

    print("\n[SUCCESS] report_gen.py is locked and final.")
