import yfinance as yf
from datetime import datetime, timedelta
import pytz
from statistics import mean
from scipy.stats import norm
import numpy as np
import pandas as pd
import statistics

class OptionStrategyCalculator:
    def __init__(self, tickers, dates, days_before, days_after, risk_free_rate, amount_per_year, puts_proportion, summary_file_name):
        self.tickers = tickers
        self.dates = dates
        self.days_before = days_before
        self.days_after = days_after
        self.risk_free_rate = risk_free_rate
        self.amount_per_year = amount_per_year
        self.puts_proportion = puts_proportion
        self.summary_file_name = summary_file_name
        self.stock_history_cache = {}
        self.volatility_cache = {}
        self.bs_cache = {}
        self.results = []
        self.yearly_changes = {}
        self.all_changes = []
        self.option_summary = {}

    def get_nearest_date(self, hist, target_date, before=True):
        if hist.empty:
            return None
        target_date = pytz.UTC.localize(target_date)
        dates = hist.index
        sorted_dates = dates.sort_values()
        candidates = [d for d in sorted_dates if (d <= target_date if before else d >= target_date)]
        return candidates[-1] if before and candidates else (candidates[0] if candidates else None)

    def get_stock_history(self, ticker, event_date):
        MAX_BEFORE_DAYS = 60
        MAX_AFTER_DAYS = 10
        key = (ticker, event_date)
        if key not in self.stock_history_cache:
            start = (event_date - timedelta(days=MAX_BEFORE_DAYS)).strftime("%Y-%m-%d")
            end = (event_date + timedelta(days=MAX_AFTER_DAYS)).strftime("%Y-%m-%d")
            stock = yf.Ticker(ticker)
            self.stock_history_cache[key] = stock.history(start=start, end=end)
        return self.stock_history_cache[key]

    def compute_volatility(self, ticker, date, hist):
        key = (ticker, date)
        if key in self.volatility_cache:
            return self.volatility_cache[key]
        hist_before = hist[hist.index < date]
        returns = hist_before['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) if not returns.empty else 0
        self.volatility_cache[key] = volatility
        return volatility

    def cached_black_scholes(self, S, K, T, r, sigma):
        key = (round(S, 2), round(K, 2), round(T, 4), round(r, 4), round(sigma, 4))
        if key in self.bs_cache:
            return self.bs_cache[key]
        if T <= 0 or sigma <= 0:
            self.bs_cache[key] = 0
            return 0
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        self.bs_cache[key] = price
        return price

    def run_strategy(self, print_out=False):
        amount_per_company = self.amount_per_year / len(self.tickers)
        if print_out:
            print(f"Amount per company: {amount_per_company:.2f}")
        for ticker in self.tickers:
            if print_out:
                print(f"Processing data for {ticker}...")
            for date in self.dates:
                self.process_trade(ticker, date, amount_per_company)
        self.output_to_csv()
        self.print_detailed_yearly(print_out)
        summary_stats = self.print_overall_summary(print_out)
        self.write_summary_stats_to_csv(*summary_stats, print_out=print_out)

    def process_trade(self, ticker, date, amount_per_company):
        window_start = date - timedelta(days=self.days_before + 45)
        window_end = date + timedelta(days=self.days_after + 5)
        hist_full = self.get_stock_history(ticker, date)
        hist = hist_full[(hist_full.index >= pd.Timestamp(window_start, tz="UTC")) & \
                        (hist_full.index <= pd.Timestamp(window_end, tz="UTC"))]
        if hist.empty or len(hist) < 30:
            return
        date_before = date - timedelta(days=self.days_before)
        date_after = date + timedelta(days=self.days_after)
        actual_before = self.get_nearest_date(hist, date_before, before=True)
        actual_after = self.get_nearest_date(hist, date_after, before=False)
        if actual_before is None or actual_after is None:
            return
        open_price = hist.loc[actual_before]["Open"]
        close_price = hist.loc[actual_after]["Close"]
        pct_change = ((close_price - open_price) / open_price) * 100
        year = date.year
        call_allocation = amount_per_company * (1 - self.puts_proportion)
        put_allocation = amount_per_company * self.puts_proportion
        hist_before = hist[hist.index < actual_before]
        volatility = self.compute_volatility(ticker, actual_before, hist)
        T = (self.days_before + self.days_after) / 252
        strike = open_price
        call_cost = self.cached_black_scholes(open_price, strike, T, self.risk_free_rate, volatility)
        contract_cost = call_cost * 100
        put_cost = self.cached_black_scholes(open_price, strike, T, self.risk_free_rate, volatility)
        put_contract_cost = put_cost * 100
        num_call_contracts = int(call_allocation // contract_cost) if contract_cost > 0 else 0
        num_put_contracts = int(put_allocation // put_contract_cost) if put_contract_cost > 0 else 0
        if num_call_contracts == 0 or num_put_contracts == 0:
            self.update_results_and_summary(year, ticker, date, actual_before, actual_after, open_price, close_price, pct_change, volatility, call_cost, 0, 0, 0, put_cost, 0, 0, 0, 0, 0)
            return
        total_cost = num_call_contracts * contract_cost
        call_value = max(0, close_price - strike)
        total_value = num_call_contracts * call_value * 100
        call_profit = total_value - total_cost
        total_put_cost = num_put_contracts * put_contract_cost
        put_value = max(0, strike - close_price)
        total_put_value = num_put_contracts * put_value * 100
        put_profit = total_put_value - total_put_cost
        self.update_results_and_summary(year, ticker, date, actual_before, actual_after, open_price, close_price, pct_change, volatility, call_cost, num_call_contracts, call_value, total_cost, call_profit, put_cost, num_put_contracts, put_value, total_put_cost, put_profit)

    def update_results_and_summary(self, year, ticker, date, actual_before, actual_after, open_price, close_price, pct_change, volatility, call_cost, num_call_contracts, call_value, total_cost, call_profit, put_cost, num_put_contracts, put_value, total_put_cost, put_profit):
        self.option_summary.setdefault(year, {"profit": 0, "trades": 0, "cost": 0, "value": 0, "profits_list": []})
        if num_call_contracts > 0:
            self.option_summary[year]["profit"] += call_profit
            self.option_summary[year]["trades"] += 1
            self.option_summary[year]["cost"] += total_cost
            self.option_summary[year]["value"] += num_call_contracts * call_value * 100
        if num_put_contracts > 0:
            self.option_summary[year]["profit"] += put_profit
            self.option_summary[year]["cost"] += total_put_cost
            self.option_summary[year]["value"] += num_put_contracts * put_value * 100
        self.all_changes.append(pct_change)
        self.yearly_changes.setdefault(year, []).append(pct_change)
        self.results.append({
            "year": year,
            "ticker": ticker,
            "target_date": date.strftime("%m-%d-%Y"),
            "before_date": actual_before.date(),
            "after_date": actual_after.date(),
            "open": round(open_price, 2),
            "close": round(close_price, 2),
            "change_pct": round(pct_change, 2),
            "volatility": round(volatility, 4),
            "call_cost_per_contract": round(call_cost, 2),
            "contracts": num_call_contracts,
            "call_value": round(call_value, 2),
            "total_cost": round(total_cost, 2),
            "profit": round(call_profit, 2),
            "put_cost_per_contract": round(put_cost, 2),
            "put_contracts": num_put_contracts,
            "put_value": round(put_value, 2),
            "put_total_cost": round(total_put_cost, 2),
            "put_profit": round(put_profit, 2)
        })

    def output_to_csv(self):
        df = pd.DataFrame(self.results)
        df.to_csv("call_put_option_strategy_results.csv", index=False)

    def print_detailed_yearly(self, print_out):
        self.results.sort(key=lambda x: -x["year"])
        current_year = None
        if print_out:
            print("\n\nDetailed Year by Year")
            for entry in self.results:
                if entry["year"] != current_year:
                    if current_year is not None:
                        avg = mean(self.yearly_changes[current_year])
                        opt = self.option_summary[current_year]
                        print(f"--> Avg change: {avg:.2f}%")
                        print(f"--> Trades: {opt['trades']}, Total cost: ${opt['cost']:.2f}, Total value: ${opt['value']:.2f}, Net profit: ${opt['profit']:.2f}")
                    current_year = entry["year"]
                    print(f"\n====== {current_year} ======")
                print(f"{entry['ticker']} | Target: {entry['target_date']} | "
                    f"Open: ${entry['open']} | Close: ${entry['close']} | "
                    f"Change: {entry['change_pct']}%")
                print(f"  Call → Contracts: {entry['contracts']:.4f} | Cost: ${entry['total_cost']:.2f} | "
                    f"Value: ${entry['call_value']:.2f} | Profit: ${entry['profit']:.2f}")
                if "put_contracts" in entry:
                    print(f"  Put  → Contracts: {entry['put_contracts']:.4f} | Cost: ${entry['put_total_cost']:.2f} | "
                        f"Value: ${entry['put_value']:.2f} | Profit: ${entry['put_profit']:.2f}")
            if current_year is not None:
                avg = mean(self.yearly_changes[current_year])
                opt = self.option_summary[current_year]
                print(f"--> Avg change: {avg:.2f}%")
                print(f"--> Trades: {opt['trades']}, Total cost: ${opt['cost']:.2f}, Total value: ${opt['value']:.2f}, Net profit: ${opt['profit']:.2f}")

    def print_overall_summary(self, print_out):
        all_profits = []
        normalized_profits = []
        if print_out:
            print("\n\nProfit by year: ")
        for y in self.option_summary.keys():
            if print_out:
                print(f"{y} | ${self.option_summary[y]["profit"]:.2f}")
            all_profits.append(self.option_summary[y]["profit"])
            normalized_profits.append(self.option_summary[y]["profit"] / self.amount_per_year)
        overall_stdev = statistics.stdev(all_profits) if len(all_profits) > 1 else 0
        normalized_stdev = statistics.stdev(normalized_profits) if len(normalized_profits) > 1 else 0
        overall_avg = mean(self.all_changes) if self.all_changes else 0
        total_profit = sum(y['profit'] for y in self.option_summary.values())
        total_trades = sum(y['trades'] for y in self.option_summary.values())
        total_cost = sum(y['cost'] for y in self.option_summary.values())
        total_value = sum(y['value'] for y in self.option_summary.values())
        median_profit = statistics.median(y['profit'] for y in self.option_summary.values())
        avg_profit = total_profit / (total_cost / self.amount_per_year) if total_cost else 0
        cv = overall_stdev / avg_profit if avg_profit else 0
        if print_out:
            print(f"\n====== OVERALL SUMMARY ======")
            print(f"Average change across all years: {overall_avg:.2f}%")
            print(f"Total trades: {total_trades}")
            print(f"Total cost: ${total_cost:.2f}")
            print(f"Total value: ${total_value:.2f}")
            print(f"Total net profit: ${total_profit:.2f}")
            print(f"\nStd dev of annual profit: ${overall_stdev:.2f}")
            print(f"Std dev of annual profit normalized: {normalized_stdev:.2f}")
            print(f"Coefficient of Variation (CV): {cv:.2f}")
            print("\nStatistics using average:")
            print(f"Average profit per year: ${avg_profit:.2f}")
            print(f"Average percentage return per year: {100 * (avg_profit / self.amount_per_year):.2f}%")
            print("\nStatistics using median:")
            print(f"Median profit per year: ${median_profit:.2f}")
            print(f"Median percentage return per year (approx): {100 * (median_profit / self.amount_per_year):.2f}%")
        return total_trades, total_cost, total_value, total_profit, avg_profit, median_profit, overall_stdev, normalized_stdev, cv, overall_avg

    def write_summary_stats_to_csv(self, total_trades, total_cost, total_value, total_profit, avg_profit, median_profit, overall_stdev, normalized_stdev, cv, overall_avg, print_out=False):
        summary_data = {
            "Total Trades": total_trades,
            "Total Cost": round(total_cost, 2),
            "Total Value": round(total_value, 2),
            "Total Profit": round(total_profit, 2),
            "Average Profit Per Year": round(avg_profit, 2),
            "Average Return % Per Year": round(100 * (avg_profit / self.amount_per_year), 2) if self.amount_per_year else 0,
            "Median Profit Per Year": round(median_profit, 2),
            "Median Return % Per Year": round(100 * (median_profit / self.amount_per_year), 2) if self.amount_per_year else 0,
            "Std Dev of Annual Profit ($)": round(overall_stdev, 2),
            "Normalized Std Dev of Profit": round(normalized_stdev, 4),
            "Coefficient of Variation": round(cv, 4),
            "Average Change %": round(overall_avg, 2),
            "Puts Proportion": self.puts_proportion,
            "Amount Per Year": self.amount_per_year,
            "Days Before": self.days_before,
            "Days After": self.days_after,
            "Risk-Free Rate": self.risk_free_rate,
            "Tickers Count": len(self.tickers),
            "Run Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        summary_df = pd.DataFrame([summary_data])
        summary_file = self.summary_file_name
        try:
            existing = pd.read_csv(summary_file)
            summary_df = pd.concat([existing, summary_df], ignore_index=True)
        except FileNotFoundError:
            pass
        summary_df.to_csv(summary_file, index=False)
        if print_out:
            print(f"\nSummary written to '{summary_file}'")
