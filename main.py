
from option_strategy_calculator import OptionStrategyCalculator
import itertools
from datetime import datetime

# === CONFIGURATION ===
DAYS_BEFORE_LIST = [2, 5, 10]  # Example values, edit as needed
DAYS_AFTER_LIST = [2, 5, 10]   # Example values, edit as needed
PUTS_PROPORTION_LIST = [0.2, 0.5, 0.8]  # Example values, edit as needed
RISK_FREE_RATE = 0.05    # 5% annualized risk-free rate
AMOUNT_PER_YEAR = 5000

# === INPUT FILES ===
with open("inputs/tickers.txt", "r") as file:
    tickers = [line.strip() for line in file if line.strip()]

with open("inputs/dates.txt", "r") as file:
    date_strs = [line.strip() for line in file if line.strip()]

dates = [datetime.strptime(d, "%m-%d-%Y") for d in date_strs]

if __name__ == "__main__":
    for days_before, days_after, puts_proportion in itertools.product(DAYS_BEFORE_LIST, DAYS_AFTER_LIST, PUTS_PROPORTION_LIST):
        summary_file_name = f"outputs/summary_db{days_before}_da{days_after}_pp{int(puts_proportion*100)}.csv"
        print(f"Running: days_before={days_before}, days_after={days_after}, puts_proportion={puts_proportion}")
        calculator = OptionStrategyCalculator(
            tickers=tickers,
            dates=dates,
            days_before=days_before,
            days_after=days_after,
            risk_free_rate=RISK_FREE_RATE,
            amount_per_year=AMOUNT_PER_YEAR,
            puts_proportion=puts_proportion,
            summary_file_name=summary_file_name
        )
        calculator.run_strategy(print_out=False)
