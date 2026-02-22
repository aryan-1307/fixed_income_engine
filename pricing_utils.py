import numpy as np
import pandas as pd
from datetime import datetime
import math

class BondAnalytics:
    @staticmethod
    def generate_cash_flows(principal, coupon_rate, frequency, years_to_maturity):
        total_periods = int(round(years_to_maturity * frequency))
        coupon_payment = (coupon_rate / frequency) * principal
        cash_flows = []

        for period in range(1, total_periods + 1):
            time = period / frequency
            amount = coupon_payment
            if period == total_periods:
                amount += principal
            cash_flows.append({'time': time, 'amount': amount})

        return cash_flows

    def calculate_price(self, yield_to_maturity, cash_flows, frequency):
        if yield_to_maturity <= -1.0:  # Logic floor for yields
            return float('inf')

        price = 0.0
        for cf in cash_flows:
            # Consistent compounding with the bond's own frequency
            price += cf['amount'] / (1 + yield_to_maturity / frequency) ** (frequency * cf['time'])
        return price

    def solve_ytm(self, target_price, cash_flows, frequency, guess=0.05):
        low = -0.20  # -20% floor
        high = 1.0   # 100% cap
        ytm = guess

        for i in range(100):
            price = self.calculate_price(ytm, cash_flows, frequency)

            # Numerical derivative: dP/dy
            eps = 1e-7
            price_eps = self.calculate_price(ytm + eps, cash_flows, frequency)
            derivative = (price_eps - price) / eps

            # Safety: If derivative is near zero or NaN, switch to bisection
            if abs(derivative) < 1e-12 or math.isnan(derivative):
                # Simple bisection step
                mid = (low + high) / 2
                ytm = mid
            else:
                diff = target_price - price
                if abs(diff) < 1e-8:
                    return ytm

                # Newton step
                ytm = ytm + diff / derivative

            # Keep solver within economic bounds
            ytm = max(min(ytm, high), low)

        return ytm

    def calculate_macaulay_duration(self, yield_to_maturity, cash_flows, frequency):
        price = self.calculate_price(yield_to_maturity, cash_flows, frequency)
        if price <= 0:
            return 0

        weighted_time = 0.0
        for cf in cash_flows:
            pv = cf['amount'] / (1 + yield_to_maturity / frequency) ** (frequency * cf['time'])
            weighted_time += cf['time'] * pv

        return weighted_time / price

    def calculate_modified_duration(self, yield_to_maturity, macaulay_duration, frequency):
        return macaulay_duration / (1 + yield_to_maturity / frequency)

    def calculate_convexity(self, yield_to_maturity, cash_flows, frequency):
        price = self.calculate_price(yield_to_maturity, cash_flows, frequency)
        if price <= 0:
            return 0

        convexity_sum = 0.0
        for cf in cash_flows:
            pv = cf['amount'] / (1 + yield_to_maturity / frequency) ** (frequency * cf['time'])
            convexity_sum += pv * (cf['time'] ** 2 + cf['time'] / frequency)

        return convexity_sum / (price * (1 + yield_to_maturity / frequency) ** 2)

    def calculate_dv01(self, yield_to_maturity, cash_flows, frequency):
        p_down = self.calculate_price(yield_to_maturity - 0.00005, cash_flows, frequency)
        p_up = self.calculate_price(yield_to_maturity + 0.00005, cash_flows, frequency)
        return (p_down - p_up) / 1.0  # Normalized for 1bp total spread

    def get_accrued_interest(self, principal, coupon_rate, frequency, last_coupon_date, settle_date):
        days_since_last = (settle_date - last_coupon_date).days
        # Standardized accrued = (Annual Coupon / Frequency) * (Days Since / Days in Period)
        # Here we use ACT/360 for the annual fraction.
        annual_coupon_amt = principal * coupon_rate
        accrued = annual_coupon_amt * (days_since_last / 360.0)
        return accrued

    def price_to_risk_metrics(self, price, principal, coupon_rate, frequency, years_to_maturity):
        cfs = self.generate_cash_flows(principal, coupon_rate, frequency, years_to_maturity)
        ytm = self.solve_ytm(price, cfs, frequency)
        macd = self.calculate_macaulay_duration(ytm, cfs, frequency)
        modd = self.calculate_modified_duration(ytm, macd, frequency)
        conv = self.calculate_convexity(ytm, cfs, frequency)

        return {
            'ytm': ytm,
            'macaulay_duration': macd,
            'modified_duration': modd,
            'convexity': conv,
            'dv01': self.calculate_dv01(ytm, cfs, frequency)
        }

if __name__ == "__main__":
    engine = BondAnalytics()

    # Test a Quarterly Pay Bond (Frequency = 4)
    bond_principal = 1000.0
    bond_coupon = 0.04
    bond_freq = 4
    bond_maturity = 10.0
    market_yield = 0.035

    cash_flow_schedule = engine.generate_cash_flows(bond_principal, bond_coupon, bond_freq, bond_maturity)
    bond_price = engine.calculate_price(market_yield, cash_flow_schedule, bond_freq)

    metrics = engine.price_to_risk_metrics(bond_price, bond_principal, bond_coupon, bond_freq, bond_maturity)

    print("--- REFINED PRICING UTILS SELF-TEST ---")
    print(f"Test Asset: 10Y Bond, 4% Coupon, Quarterly Pay, 3.5% Yield")
    print(f"Calculated Price:    {bond_price:.4f}")
    print(f"Verified YTM:        {metrics['ytm']:.4%}")
    print(f"Modified Duration:   {metrics['modified_duration']:.4f}")
    print(f"Convexity:           {metrics['convexity']:.4f}")
    print(f"DV01:                {metrics['dv01']:.4f}")

    # Accrued Interest Test
    last_cpn = datetime(2025, 12, 1)
    settle = datetime(2026, 2, 15)
    accrued = engine.get_accrued_interest(bond_principal, bond_coupon, bond_freq, last_cpn, settle)
    print(f"Accrued Interest:    {accrued:.4f}")

    assert abs(metrics['ytm'] - market_yield) < 1e-7, "YTM Solver Inaccuracy"
    print("\n[SUCCESS] pricing_utils.py is now locked and ready.")
