"""
Project: InvestiBuddy
Last Updated: 09/03/25 11:15PM
Final Version
"""
# -------------------- Import --------------------
import os
import csv
import sqlite3
import pandas as pd
import datetime
import yfinance as yf
from typing import List, Dict, Any, Optional, Tuple

# -------------------- Database Setup --------------------
def create_database():
    conn = sqlite3.connect("portfolio_manager.db") #saving to portfolio_manager.db as multiple tables
    cursor = conn.cursor()

    # Create Users Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        user_id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        risk_tolerance TEXT CHECK(risk_tolerance IN ('Low', 'Medium', 'High')) NOT NULL
    )
    """)

    # Create Portfolios Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS portfolios (
        portfolio_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        name TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users (user_id)
    )
    """)

    # Create Symbols Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS symbols (
        symbol_id INTEGER PRIMARY KEY AUTOINCREMENT,
        portfolio_id INTEGER NOT NULL,
        ticker TEXT NOT NULL,
        sector TEXT,
        FOREIGN KEY (portfolio_id) REFERENCES portfolios (portfolio_id)
    )
    """)

    # Create Transactions Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS transactions (
        transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol_id INTEGER NOT NULL,
        transaction_type TEXT NOT NULL,
        shares REAL NOT NULL,
        price REAL NOT NULL,
        transaction_cost REAL NOT NULL,
        transaction_date TEXT NOT NULL,
        FOREIGN KEY (symbol_id) REFERENCES symbols (symbol_id)
    )
    """)

    conn.commit()
    conn.close()

# -------------------- Yahoo Finance API Data Fetching --------------------
class YFinanceDataSource:
    def fetch_data(self, ticker: str, period: str = "1d") -> dict:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            history = stock.history(period=period)

            last_price = info.get("currentPrice", None)
            if last_price is None:  # Use previous close if current price is unavailable
                last_price = info.get("previousClose", None)
            if last_price is None and not history.empty:  # If both missing, use historical data
                last_price = history["Close"].iloc[-1]
            if last_price is None:  # If still no price found
                return {"error": f"⚠️ No price data found for {ticker}"}

            return {
                "ticker": ticker,
                "company_name": info.get("shortName", "N/A"),
                "sector": info.get("sector", "N/A"),
                "last_price": last_price,
                "change": last_price - info.get("previousClose", last_price),
                "change_percent": ((last_price - info.get("previousClose", last_price)) / info.get("previousClose", last_price)) * 100 if info.get("previousClose") else 0,
                "market_time": history.index[-1].strftime("%Y-%m-%d %H:%M:%S") if not history.empty else "N/A",
                "market_cap": info.get("marketCap", "N/A"),
                "volume": history["Volume"].iloc[-1] if not history.empty else "N/A",
                "high": history["High"].iloc[-1] if not history.empty else "N/A",
                "low": history["Low"].iloc[-1] if not history.empty else "N/A",
                "open": history["Open"].iloc[-1] if not history.empty else "N/A",
                "previous_close": info.get("previousClose", last_price),
            }

        except Exception as e:
            return {"error": f"⚠️ Error fetching data for {ticker}: {str(e)}"}

# -------------------- Database Manager --------------------
class DatabaseManager:
    def __init__(self, db_name: str = "portfolio_manager.db"):
        self.db_name = db_name

    # Establish database connection before query can be run
    def get_connection(self):
        return sqlite3.connect(self.db_name)

    # Executing a SELECT query and returning results
    def execute_query(self, query: str, params: tuple = None) -> list:
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(query, params if params else ())
            result = cursor.fetchall()
            return result
        except sqlite3.OperationalError as e:
            print(f"⚠️ Database error: {e}")
            return []
        finally:
            conn.close()

    # Executing an action query (INSERT, UPDATE, DELETE) and returning last row ID
    def execute_action(self, query: str, params: tuple = None) -> int:
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(query, params if params else ())
            last_id = cursor.lastrowid
            conn.commit()
            return last_id
        except sqlite3.OperationalError as e:
            print(f"⚠️ Database error: {e}")
            conn.rollback()
            return -1
        finally:
            conn.close()

# -------------------- User Authentication --------------------
# User registration, login, and management
class UserManager:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    # User registration
    def register_user(self, username: str, password: str, risk_tolerance: str) -> bool:
        try:
            self.db_manager.execute_action(
                "INSERT INTO users (username, password, risk_tolerance) VALUES (?, ?, ?)",
                (username, password, risk_tolerance))
            return True
        except sqlite3.IntegrityError:
            return False  # Username already exists

    # User login
    def login_user(self, username: str, password: str) -> Optional[int]:
        result = self.db_manager.execute_query(
            "SELECT user_id FROM users WHERE username = ? AND password = ?",
            (username, password))
        if result:
            return result[0][0]
        return None

    # Fetch risk tolerance
    def get_user_risk_tolerance(self, user_id: int) -> str:
        result = self.db_manager.execute_query( #fetch user's risk tolerance from db
            "SELECT risk_tolerance FROM users WHERE user_id = ?", (user_id,))
        return result[0][0] if result else "Low"  # if unknown, default= Conservative

    # Update risk tolerance
    def change_risk_tolerance(self, user_id: int):
        print("\n===== Change Risk Tolerance =====")
        print("1. Low Risk (Max 20% per sector - Safer allocation)")
        print("2. Medium Risk (Max 30% per sector - Balanced)")
        print("3. High Risk (Max 40% per sector - Aggressive growth)")
        choice = input("Enter your new risk tolerance (1, 2, or 3): ").strip()

        if choice == "1":
            new_risk_tolerance = "Low"
        elif choice == "2":
            new_risk_tolerance = "Medium"
        elif choice == "3":
            new_risk_tolerance = "High"
        else:
            print("❌ Invalid input. Risk tolerance not changed.")
            return

        self.db_manager.execute_action(
            "UPDATE users SET risk_tolerance = ? WHERE user_id = ?",
            (new_risk_tolerance, user_id))
        print(f"✅ Your risk tolerance has been updated to: {new_risk_tolerance}")

# -------------------- Portfolio & Stock Management --------------------

# Representing a user's portfolio
class Portfolio:
    def __init__(self, portfolio_id: int, user_id: int, name: str):
        self.portfolio_id = portfolio_id
        self.user_id = user_id
        self.name = name
        self.symbols = []

# Stocks
class Symbol:
    def __init__(self, symbol_id: int, portfolio_id: int, ticker: str, sector: str):
        self.symbol_id = symbol_id
        self.portfolio_id = portfolio_id
        self.ticker = ticker
        self.sector = sector
        self.transactions = []
        self.current_data = None

# Transactions
class Transaction:
    def __init__(self, transaction_id: int, symbol_id: int, transaction_type: str, shares: float, price: float, transaction_cost: float, transaction_date: str):
        self.transaction_id = transaction_id
        self.symbol_id = symbol_id
        self.transaction_type = transaction_type
        self.shares = shares
        self.price = price
        self.transaction_cost = transaction_cost
        self.transaction_date = transaction_date

# -------------------- Portfolio Manager --------------------
class PortfolioManager:
    def __init__(self, db_manager: DatabaseManager): #initialisation
        self.db_manager = db_manager
        self.yfinance_source = YFinanceDataSource()

    def get_portfolio_symbols(self, portfolio_id: int) -> List[Symbol]: #fetch all symbols in given portfolio
        results = self.db_manager.execute_query(
            "SELECT symbol_id, portfolio_id, ticker, sector FROM symbols WHERE portfolio_id = ?",
            (portfolio_id,))

        symbols = []
        for row in results:
            symbol = Symbol(symbol_id=row[0],portfolio_id=row[1],ticker=row[2],sector=row[3])
            symbol.current_data = self.yfinance_source.fetch_data(symbol.ticker) #fetch latest stock data
            symbol.transactions = self.get_symbol_transactions(symbol.symbol_id) #fetch all transations
            symbols.append(symbol)
        return symbols

    # Creating a new portfolio for user
    def create_portfolio(self, user_id: int, name: str) -> int:
        portfolio_id = self.db_manager.execute_action(
            "INSERT INTO portfolios (user_id, name) VALUES (?, ?)",
            (user_id, name))
        return portfolio_id

    # Getting all portfolios for user
    def get_user_portfolios(self, user_id: int) -> List[Portfolio]:
        results = self.db_manager.execute_query(
            "SELECT portfolio_id, user_id, name FROM portfolios WHERE user_id = ?",
            (user_id,)
        )
        portfolios = []
        for row in results:
            portfolio = Portfolio(portfolio_id=row[0],user_id=row[1],name=row[2])
            portfolios.append(portfolio)
        return portfolios

    # Adding symbol to a portfolio
    def add_symbol(self, portfolio_id: int, ticker: str, sector: str) -> Optional[int]:

        existing_symbol = self.db_manager.execute_query( #check if the symbol is currently already added to the portfolio
            "SELECT symbol_id FROM symbols WHERE portfolio_id = ? AND ticker = ?",
            (portfolio_id, ticker))
        if existing_symbol:
            print(f"⚠️ Symbol '{ticker}' is already in the portfolio. Cannot add duplicate.")
            return None

        symbol_id = self.db_manager.execute_action( #if symbol has not been added, add new symbol
            "INSERT INTO symbols (portfolio_id, ticker, sector) VALUES (?, ?, ?)",
            (portfolio_id, ticker, sector))
        print(f"✅ Symbol '{ticker}' added successfully!")
        return symbol_id

    # Getting all symbols in a portfolio
    def add_transaction(self, symbol_id: int, transaction_type: str, shares: float, price: float,
                        transaction_cost: float, transaction_date: str) -> int:

        result = self.db_manager.execute_query(
            "SELECT ticker FROM symbols WHERE symbol_id = ?", (symbol_id,))
        if not result:
            print("❌ Symbol not found.")
            return -1  # Exit with error code
        ticker = result[0][0]  # Extract the ticker

        # If sell, check if user has enough shares first
        if transaction_type.lower() == "sell":

            # for specific symbol, calculate total bought
            result = self.db_manager.execute_query(
                "SELECT SUM(shares) FROM transactions WHERE symbol_id = ? AND LOWER(transaction_type) = 'buy'",
                (symbol_id,))
            total_bought = result[0][0] if result and result[0][0] else 0

            # for specific symbol, calculate total sold
            result = self.db_manager.execute_query(
                "SELECT SUM(shares) FROM transactions WHERE symbol_id = ? AND LOWER(transaction_type) = 'sell'",
                (symbol_id,))
            total_sold = result[0][0] if result and result[0][0] else 0

            available_shares = total_bought - total_sold

            if shares > available_shares:
                print(f"❌ Cannot sell {shares} shares. You only own {available_shares} shares.")
                return -1  # stop without adding transaction

        transaction_id = self.db_manager.execute_action( #add transaction if there is enough shares to sell
            "INSERT INTO transactions (symbol_id, transaction_type, shares, price, transaction_cost, transaction_date) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (symbol_id, transaction_type, shares, price, transaction_cost, transaction_date))
        return transaction_id

    # Getting all transactions for a symbol
    def get_symbol_transactions(self, symbol_id: int) -> List[Transaction]:
        results = self.db_manager.execute_query(
            "SELECT transaction_id, symbol_id, transaction_type, shares, price, transaction_cost, transaction_date FROM transactions WHERE symbol_id = ?",
            (symbol_id,))
        transactions = []
        for row in results:
            transaction = Transaction(
                transaction_id=row[0],
                symbol_id=row[1],
                transaction_type=row[2],
                shares=row[3],
                price=row[4],
                transaction_cost=row[5],
                transaction_date=row[6]
            )
            transactions.append(transaction)
        return transactions

    # Calculating metrics for a portfolio
    def calculate_portfolio_metrics(self, portfolio_id: int) -> Dict:
        symbols = self.get_portfolio_symbols(portfolio_id)
        total_investment = 0
        total_current_value = 0
        total_realised_pl = 0
        unrealised_pl_percent = 0
        symbol_metrics = []

        # Calculating metrics for each symbol
        for symbol in symbols:
            symbol_metric = self.calculate_symbol_metrics(symbol)
            symbol_metrics.append(symbol_metric)
            total_investment += symbol_metric["total_investment"]
            total_current_value += symbol_metric["current_value"]
            total_realised_pl += symbol_metric["realised_pl"]

        active_investment = sum(symbol["current_shares"] * symbol["avg_cost"] for symbol in symbol_metrics if
                                symbol["current_shares"] > 0)
        if active_investment > 0:
            unrealised_pl_percent = ((total_current_value - active_investment) / active_investment * 100)

        portfolio_metrics = {
            "total_investment": total_investment,
            "total_current_value": total_current_value,
            "total_unrealised_pl": total_current_value - active_investment,
            "total_unrealised_pl_percent": unrealised_pl_percent,
            "total_realised_pl": total_realised_pl,
            "symbols": symbol_metrics
        }
        return portfolio_metrics

    # Calculating metrics for a symbol
    def calculate_symbol_metrics(self, symbol: Symbol) -> Dict:
        current_price = symbol.current_data.get("last_price", 0) if symbol.current_data and "error" not in symbol.current_data else 0
        total_shares = 0
        total_investment = 0
        total_sold_amount = 0
        total_sold_shares = 0

        for transaction in symbol.transactions:
            if transaction.transaction_type.lower() == "buy":
                total_shares += transaction.shares
                total_investment += (transaction.shares * transaction.price) + transaction.transaction_cost
            elif transaction.transaction_type.lower() == "sell":
                total_sold_shares += transaction.shares
                total_sold_amount += (transaction.shares * transaction.price) - transaction.transaction_cost

        # Current holdings
        current_shares = total_shares - total_sold_shares
        current_shares = max(current_shares, 0)

        # Calculate average cost basis
        avg_cost = (total_investment / total_shares) if total_shares > 0 else 0

        # Reset avg_cost to 0 if no current shares
        avg_cost = avg_cost if current_shares > 0 else 0

        # Calculate realised P/L from sales
        realised_pl = 0
        if total_sold_shares > 0:
            avg_cost_per_share = total_investment / total_shares if total_shares > 0 else 0
            realised_pl = total_sold_amount - (total_sold_shares * avg_cost_per_share)

        # Calculate current value and unrealised P/L
        current_value = current_shares * current_price
        unrealised_pl = current_value - (current_shares * avg_cost)

        symbol_metrics = {
            "ticker": symbol.ticker,
            "sector": symbol.sector,
            "current_price": current_price,
            "avg_cost": avg_cost,
            "current_shares": current_shares,
            "total_investment": total_investment,
            "realised_pl": realised_pl,
            "current_value": current_value,
            "unrealised_pl": unrealised_pl,
            "unrealised_pl_percent": (unrealised_pl / (
                        current_shares * avg_cost) * 100) if current_shares > 0 and avg_cost > 0 else 0,
            "day_change": symbol.current_data.get("change",
                                                  0) if symbol.current_data and "error" not in symbol.current_data else 0,
            "day_change_percent": symbol.current_data.get("change_percent",
                                                          0) if symbol.current_data and "error" not in symbol.current_data else 0
        }
        return symbol_metrics
    # Sector exposure analysis of the user's portfolio
    def calculate_sector_exposure(self, portfolio_id: int, user_id: int, silent: bool = False) -> Dict:
        """Calculating sector exposure to return results such that if silent=True, it should not print."""
        symbols = self.get_portfolio_symbols(portfolio_id)
        total_value = sum(self.calculate_symbol_metrics(symbol)["current_value"] for symbol in symbols)

        if total_value == 0:
            if not silent:
                print("⚠️ No stocks in portfolio to analyze.")
            return {}

        sector_distribution = {}

        # Calculate sector exposure distribution
        for symbol in symbols:
            value = self.calculate_symbol_metrics(symbol)["current_value"]
            sector_distribution[symbol.sector] = sector_distribution.get(symbol.sector, 0) + value

        # Converting raw values to %
        for sector in sector_distribution:
            sector_distribution[sector] = {
                "value": sector_distribution[sector],
                "percentage": (sector_distribution[sector] / total_value) * 100
            }

        if not silent:
            print("\n📊 Sector Exposure Analysis:")
            for sector, data in sector_distribution.items():
                print(f"{sector}: ${data['value']:.2f} ({data['percentage']:.2f}%)")

        return sector_distribution

    # Threshold based rebalancing fixed at 20%, 30%, 40%
    def suggest_rebalancing(self, portfolio_id: int, user_id: int, threshold: float) -> Dict:
        sector_exposure = self.calculate_sector_exposure(portfolio_id, user_id, silent=True)
        symbols = self.get_portfolio_symbols(portfolio_id)

        if not symbols:
            print("⚠️ No stocks in portfolio to analyse.")
            return {}

        if len(symbols) == 1:
            print("⚠️ Your portfolio has only one stock. Rebalancing is not possible.")
            print("📌 Consider adding more stocks to diversify.")
            return {}

        # Identify Overweight & Underweight Sectors
        over_exposed_sectors = {}
        under_exposed_sectors = {}

        for sector, data in sector_exposure.items():
            if data["percentage"] > threshold:
                over_exposed_sectors[sector] = data
            elif data["percentage"] < threshold:
                under_exposed_sectors[sector] = data

        # Display Sector Exposure for Transparency
        print("\n📊 Sector Exposure Analysis:")
        for sector, data in sector_exposure.items():
            print(f"- {sector}: ${data['value']:.2f} ({data['percentage']:.2f}%)")

        if not over_exposed_sectors:
            print("✅ Your portfolio is well-balanced. No rebalancing needed.")
            return {}

        print("\n⚠️ Rebalancing Needed: The following sectors exceed your risk threshold:")
        for sector, data in over_exposed_sectors.items():
            print(f"- {sector}: {data['percentage']:.2f}% (Exceeds {threshold:.1f}%)")

        print("\nOptions:")
        print("1️⃣ Auto-Rebalance (Sell overweight stocks, buy underweight ones)")
        print("2️⃣ Skip for now")

        choice = input("Choose an option (1 or 2): ").strip()
        if choice == "1":
            self.auto_rebalance(portfolio_id, user_id, over_exposed_sectors, under_exposed_sectors, threshold)
        else:
            print("✅ Rebalancing skipped for now.")

        return {"over_exposed": over_exposed_sectors, "under_exposed": under_exposed_sectors}

    def get_risk_tolerance_threshold(self, user_id: int) -> float:
        """Fetch the user's risk tolerance and return the corresponding max sector allocation threshold."""
        result = self.db_manager.execute_query(
            "SELECT risk_tolerance FROM users WHERE user_id = ?", (user_id,)
        )

        # Default to 'Low' risk if user data is not found
        risk_tolerance = result[0][0] if result else "Low"
        thresholds = {"Low": 20.0, "Medium": 30.0, "High": 40.0}

        return thresholds.get(risk_tolerance, 20.0)  # Default to 20% (Low Risk)

    # Auto-Rebalancing
    def auto_rebalance(self, portfolio_id: int, user_id: int, over_exposed_sectors: Dict, under_exposed_sectors: Dict,
                       threshold: float) -> None:
        print("\n🔄 Auto-Rebalancing in Progress...")

        # Get fresh data for each rebalancing run
        symbols = self.get_portfolio_symbols(portfolio_id)

        if len(symbols) <= 1:
            print("⚠️ Cannot rebalance. Your portfolio has only one stock.")
            print("📌 Consider adding more stocks to diversify.")
            return

        # Recalculate current sector exposure to ensure we're using fresh data
        current_exposure = self.calculate_sector_exposure(portfolio_id, user_id, silent=True)

        # Recalculate overweight/underweight sectors based on current data
        over_exposed_sectors = {}
        under_exposed_sectors = {}

        total_value = sum(data["value"] for sector, data in current_exposure.items())

        for sector, data in current_exposure.items():
            if data["percentage"] > threshold:
                over_exposed_sectors[sector] = data
            elif data["percentage"] < threshold:
                under_exposed_sectors[sector] = data

        if not over_exposed_sectors:
            print("✅ Your portfolio is already well-balanced. No rebalancing needed.")
            return

        # Step 1: Sell from Overweight Sectors
        total_funds_to_reallocate = 0

        for sector, data in over_exposed_sectors.items():
            excess_percentage = data["percentage"] - threshold
            sector_value = data["value"]
            target_value = (threshold / 100) * total_value  # Calculate target based on total portfolio value
            sell_amount = sector_value - target_value

            print(f"🔹 Selling from {sector}: Need to reduce by ${sell_amount:.2f}")

            # Identify tradable stocks in this sector
            sector_symbols = [
                s for s in symbols if s.sector == sector and self.calculate_symbol_metrics(s)["current_shares"] > 0
            ]

            if not sector_symbols:
                print(f"⚠️ No tradable stocks found in {sector}. Skipping.")
                continue

            # Sort by largest holding
            sector_symbols.sort(key=lambda x: self.calculate_symbol_metrics(x)["current_value"], reverse=True)
            remaining_sell = sell_amount

            for symbol in sector_symbols:
                symbol_metrics = self.calculate_symbol_metrics(symbol)
                current_shares = symbol_metrics["current_shares"]

                # Add a safety check to never sell all shares
                max_shares_to_sell = current_shares * 0.9  # Never sell more than 90% of a position in one go

                sell_shares = round((remaining_sell / symbol_metrics["current_price"]), 2)
                sell_shares = min(sell_shares, max_shares_to_sell)

                if sell_shares > 0:
                    self.add_transaction(symbol.symbol_id, "sell", sell_shares, symbol_metrics["current_price"], 0,
                                         datetime.datetime.now().strftime("%Y-%m-%d"))
                    remaining_sell -= sell_shares * symbol_metrics["current_price"]
                    print(
                        f"✅ Sold {sell_shares:.2f} shares of {symbol.ticker} (keeping {current_shares - sell_shares:.2f} shares).")

                total_funds_to_reallocate += sell_shares * symbol_metrics["current_price"]

        # Step 2: Buy into Underweight Sectors
        if total_funds_to_reallocate > 0:
            print(f"\n🔹 Reinvesting ${total_funds_to_reallocate:.2f} into underweight sectors...")

            underweight_symbols = [s for s in symbols if s.sector in under_exposed_sectors]

            if not underweight_symbols:
                print("⚠️ No underweight sectors available for investment.")
                return

            total_underweight_percentage = sum(
                threshold - under_exposed_sectors[sector]["percentage"]
                for sector in under_exposed_sectors
            )

            for symbol in underweight_symbols:
                sector_percentage_deficit = threshold - under_exposed_sectors[symbol.sector]["percentage"]
                allocation_ratio = sector_percentage_deficit / total_underweight_percentage
                funds_for_symbol = total_funds_to_reallocate * allocation_ratio

                symbol_metrics = self.calculate_symbol_metrics(symbol)
                shares_to_buy = round(funds_for_symbol / symbol_metrics["current_price"], 2)

                if shares_to_buy > 0:
                    self.add_transaction(symbol.symbol_id, "buy", shares_to_buy, symbol_metrics["current_price"], 0,
                                         datetime.datetime.now().strftime("%Y-%m-%d"))
                    print(f"✅ Bought {shares_to_buy:.2f} shares of {symbol.ticker} in {symbol.sector} sector.")

        print("✅ Auto-Rebalancing Complete. Portfolio is now aligned with risk tolerance.")

        # Display updated portfolio after rebalancing
        print("\n📊 Updated Portfolio After Rebalancing:")
        symbols = self.get_portfolio_symbols(portfolio_id)  # Refresh symbols
        for symbol in symbols:
            metrics = self.calculate_symbol_metrics(symbol)
            if metrics["current_shares"] > 0:
                print(
                    f"- {symbol.ticker} ({symbol.sector}): {metrics['current_shares']:.2f} shares (${metrics['current_value']:.2f})")


    # Exporting portfolio data to CSV file
    def export_portfolio_to_csv(self, portfolio_id: int, user_id: int, directory: str) -> Optional[str]:
        try:
            metrics = self.calculate_portfolio_metrics(portfolio_id)

            sector_exposure = self.calculate_sector_exposure(portfolio_id, user_id, silent=True)

            # Getting portfolio details
            portfolio_result = self.db_manager.execute_query(
                "SELECT name FROM portfolios WHERE portfolio_id = ?",
                (portfolio_id,)
            )
            if not portfolio_result:
                print("❌ Portfolio not found.")
                return None

            portfolio_name = portfolio_result[0][0].replace(" ", "_")
            filename = f"{portfolio_name}_report_{datetime.datetime.now().strftime('%Y%m%d')}.csv"
            file_path = os.path.join(directory, filename)

            # To ensure that the directory exists
            os.makedirs(directory, exist_ok=True)

            # Define correct fieldnames
            fieldnames = [
                "Ticker", "Sector", "Current Price", "Avg Cost", "Shares",
                "Investment", "Current Value", "Unrealised P/L", "Unrealised P/L %",
                "Day Change", "Day Change %"
            ]

            # Open CSV file for writing
            with open(file_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                # Add symbol details
                for symbol in metrics["symbols"]:
                    writer.writerow({
                        "Ticker": symbol["ticker"],
                        "Sector": symbol.get("sector", "N/A"),
                        "Current Price": f"${symbol['current_price']:.2f}",
                        "Avg Cost": f"${symbol['avg_cost']:.2f}",
                        "Shares": f"{symbol['current_shares']:.2f}",
                        "Investment": f"${symbol['total_investment']:.2f}",
                        "Current Value": f"${symbol['current_value']:.2f}",
                        "Unrealised P/L": f"${symbol['unrealised_pl']:.2f}",
                        "Unrealised P/L %": f"{symbol['unrealised_pl_percent']:.2f}%",
                        "Day Change": f"${symbol['day_change']:.2f}",
                        "Day Change %": f"{symbol['day_change_percent']:.2f}%"
                    })

                writer.writerow({})
                writer.writerow({"Ticker": "Sector Exposure Analysis"})
                writer.writerow({"Ticker": "Sector", "Sector": "Value ($)", "Current Price": "Percentage (%)"})

                for sector, data in sector_exposure.items():
                    writer.writerow({
                        "Ticker": sector,
                        "Sector": f"${data['value']:.2f}",
                        "Current Price": f"{data['percentage']:.2f}%"
                    })

            return file_path

        except Exception as e:
            print(f"⚠️ Error exporting portfolio: {str(e)}")
            return None


# -------------------- Main Application --------------------
class MainApp:
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.user_manager = UserManager(self.db_manager)
        self.portfolio_manager = PortfolioManager(self.db_manager)
        self.current_user_id = None
        self.current_portfolio_id = None

    # Beginner's Guide
    def beginner_guide(self):
        print("\n===== 📖 Beginner's Guide: How to Use InvestiBuddy Portfolio Manager =====")
        print("\n 1. Getting Started")
        print("   - Register an account and log in.")
        print("   - Set your risk tolerance (Low, Medium, High).")

        print("\n 2. Creating a Portfolio")
        print("   - Navigate to 'Create New Portfolio'.")
        print("   - You can have multiple portfolios for different strategies.")

        print("\n 3. Adding Stocks to Your Portfolio")
        print("   - Select 'Add Symbol' to search for a stock ticker (e.g., AAPL, TSLA).")
        print("   - The app will automatically fetch stock data.")
        print("   - Each stock belongs to a sector (Technology, Consumer, etc.).")

        print("\n 4. Buying & Selling Stocks")
        print("   - Choose 'Add Transaction' to buy or sell stocks.")
        print("   - Enter the number of shares and confirm the price.")

        print("\n 5. Viewing Portfolio Performance")
        print("   - Use 'View Portfolio Summary' to track your total investment, profit/loss.")
        print("   - 'Analyse Sector Exposure' will show if you are overexposed in one sector.")

        print("\n 6. Rebalancing Your Portfolio")
        print("   - If a sector exceeds your set risk threshold (e.g., 20%), rebalancing is suggested.")
        print("   - You can auto-rebalance or manually adjust stocks.")

        print("\n🎯 Pro Tips:")
        print("✅ Diversify across multiple sectors to reduce risk.")
        print("✅ Regularly check your portfolio to stay within your risk tolerance.")
        print("✅ Use 'Export Portfolio Report' to keep records of your trades in a csv file.")

        print("\n😊 Happy Investing! You can always return to this guide anytime.")

        input("\nPress ENTER to return to the main menu.")

    # Main Menu
    def display_menu(self):
        if self.current_user_id is None:
            print("\n===== 📌 Welcome to InvestiBuddy Portfolio Manager! =====")
            print("1. View Beginner’s Guide")
            print("2. Register")
            print("3. Login")
            print("0. Exit")
        elif self.current_portfolio_id is None:
            risk_tolerance = self.user_manager.get_user_risk_tolerance(self.current_user_id)
            print(f"\n===== Portfolio Management (Risk Tolerance: {risk_tolerance}) =====")
            print("1. View Beginner’s Guide")
            print("2. Create New Portfolio")
            print("3. View My Portfolios")
            print("4. Change Risk Tolerance")
            print("5. Logout")
            print("0. Exit")
        else:
            risk_tolerance = self.user_manager.get_user_risk_tolerance(self.current_user_id)
            print(f"\n===== Portfolio Operations (Risk Tolerance: {risk_tolerance}) =====")
            print("1. View Beginner’s Guide")
            print("2. Add Symbol")
            print("3. View Symbols")
            print("4. Add Transaction")
            print("5. View Portfolio Summary")
            print("6. Analyse Sector Exposure")
            print("7. Export Portfolio Report")
            print("8. Back to Portfolios")
            print("0. Exit")

    # User Registration
    def register(self):
        print("\n===== 📝 Register =====")

        user_data = {"username": "", "password": "", "risk_tolerance": ""}
        steps = ["username", "password", "risk_tolerance"]
        current_step = 0

        while True:
            while current_step < len(steps):
                step_name = steps[current_step]

                # Show current entry (if any)
                print(
                    f"Current {step_name.capitalize()}: [{user_data[step_name] if user_data[step_name] else 'Not Set'}]")

                if step_name == "risk_tolerance":
                    print("\n📊 Select Your Risk Profile: ")
                    print("1️⃣ Low Risk (Max 20% sector exposure, safer allocation)")
                    print("2️⃣ Medium Risk (Max 30% sector exposure, balanced)")
                    print("3️⃣ High Risk (Max 40% sector exposure, aggressive growth)")

                    entry = input(
                        "Enter choice (1, 2, or 3) (Press ENTER to keep, 'B' to go back, 'X' to exit): ").strip()

                    if entry.lower() == 'b' and current_step > 0:
                        current_step -= 1
                    elif entry.lower() == 'x':
                        print("❌ Registration cancelled.")
                        return
                    elif entry == "1":
                        user_data["risk_tolerance"] = "Low"
                        current_step += 1
                    elif entry == "2":
                        user_data["risk_tolerance"] = "Medium"
                        current_step += 1
                    elif entry == "3":
                        user_data["risk_tolerance"] = "High"
                        current_step += 1
                    elif not entry and user_data["risk_tolerance"]:
                        current_step += 1
                    else:
                        print("⚠️ Invalid input. Please enter '1', '2', or '3'.")

                else:
                    entry = input(f"Enter {step_name} (Press ENTER to keep, 'B' to go back, 'X' to exit): ").strip()

                    if entry.lower() == 'b' and current_step > 0:
                        current_step -= 1
                    elif entry.lower() == 'x':
                        print("❌ Registration cancelled.")
                        return
                    elif not entry and not user_data[step_name]:
                        print(f"⚠️ {step_name.capitalize()} cannot be empty. Please enter a valid {step_name}.")
                    else:
                        if entry:
                            user_data[step_name] = entry
                        current_step += 1

                        # Confirmation Step**
            while True:
                print("\n📌 Review Your Details:")
                print(f"👤 Username: {user_data['username']}")
                print(f"🔑 Password: {'*' * len(user_data['password'])}")
                print(
                    f"📊 Risk Tolerance: {user_data['risk_tolerance']} (Sector Limit: {self.get_risk_limit(user_data['risk_tolerance'])}%)")

                confirm = input("Confirm registration? (Y = Yes, B = Back, X = Exit): ").strip().lower()

                if confirm == 'y':
                    if self.user_manager.register_user(user_data["username"], user_data["password"],
                                                       user_data["risk_tolerance"]):
                        print("✅ Account created successfully! Please log in.")
                        return
                    else:
                        print("⚠️ Username already exists. Try again.")
                        current_step = 0
                        break
                elif confirm == 'b':
                    current_step = len(steps) - 1
                    break
                elif confirm == 'x':
                    print("❌ Registration cancelled.")
                    return
                else:
                    print("⚠️ Invalid input. Please enter 'Y', 'B', or 'X'.")

    # Using a helper function to get sector limit for display
    def get_risk_limit(self, risk_tolerance: str) -> int:
        risk_thresholds = {"Low": 20, "Medium": 30, "High": 40}
        return risk_thresholds.get(risk_tolerance, 20)

    # User Login
    def login(self):
        print("\n===== Login =====")

        user_data = {"username": "", "password": ""}
        steps = ["username", "password"]  # Order of steps
        current_step = 0  # Start at the first step

        while True:
            while current_step < len(steps):
                step_name = steps[current_step]  # Get current step

                # Show current entry (if any)
                print(
                    f"Current {step_name.capitalize()}: [{user_data[step_name] if user_data[step_name] else 'Not Set'}]")
                entry = input(f"Enter {step_name} (Press ENTER to keep, 'B' to go back, 'X' to cancel login): ").strip()

                if entry.lower() == 'b':  # Backtrack to the previous step
                    if current_step == 0:
                        print("❌ Already at the first step, cannot go back.")
                    else:
                        current_step -= 1  # Move back
                elif entry.lower() == 'x':  # Cancel login
                    print("❌ Login cancelled.")
                    return  # Exit function, return to main menu
                elif entry:  # Save new entry if not empty
                    user_data[step_name] = entry
                    current_step += 1  # Move forward
                else:  # If empty, keep existing value
                    current_step += 1

            # Attempt login
            user_id = self.user_manager.login_user(user_data["username"], user_data["password"])
            if user_id:
                self.current_user_id = user_id
                print(f"😊 Welcome, {user_data['username']}!")
                return  # Exit after successful login

            # If login fails, allow backtracking or retry
            print("❌ Invalid username or password. Try again.")
            retry_choice = input("Press ENTER to retry, 'B' to go back, or 'X' to cancel login: ").strip().lower()

            if retry_choice == 'b':
                current_step = len(steps) - 1  # Move back to password step
            elif retry_choice == 'x':
                print("❌ Login cancelled.")
                return  # Exit function

    # User Logout
    def logout(self):
        self.current_user_id = None
        self.current_portfolio_id = None
        print("Logged out successfully.")
    #create new portfolio
    def create_portfolio(self):
        print("\n===== Create Portfolio =====")
        name = input("Portfolio Name: ")
        portfolio_id = self.portfolio_manager.create_portfolio(self.current_user_id, name)
        print(f"✅ Portfolio '{name}' created successfully!")

    #view portfolios of current user
    def view_portfolios(self):
        print("\n===== My Portfolios =====")
        portfolios = self.portfolio_manager.get_user_portfolios(self.current_user_id)

        if not portfolios:
            print("You don't have any portfolios yet.")
            return

        while True:
            for i, portfolio in enumerate(portfolios, 1):
                print(f"{i}. {portfolio.name}")

            choice = input("\nSelect a portfolio number (or type 'b' to go back): ")
            if choice.lower() == 'b':
                return  # Go back to the main menu
            if choice.isdigit() and 1 <= int(choice) <= len(portfolios):
                self.current_portfolio_id = portfolios[int(choice) - 1].portfolio_id
                return
            else:
                print("Invalid choice. Please try again or type 'b' to go back.")

    #add ticker symbol to current portfolio

    def add_symbol(self):
        """Allows the user to add a stock symbol to their portfolio."""
        print("\n===== Add Symbol =====")
        while True:
            ticker = input("Enter ticker symbol (or type 'b' to go back): ").upper().strip()
            if ticker.lower() == 'b':
                return  # Go back to the portfolio menu

            yf_source = YFinanceDataSource()
            data = yf_source.fetch_data(ticker)

            if "error" in data:
                print(f"⚠️ Error: {data['error']}. Try again or type 'b' to go back.")
                continue  # Let the user retry

            sector = yf.Ticker(ticker).info.get("sector", "Unknown")
            symbol_id = self.portfolio_manager.add_symbol(self.current_portfolio_id, ticker, sector)

            if symbol_id is not None:  # If symbol was successfully added
                return

    #views all symbols added to portfolio
    def view_symbols(self):
        print("\n===== Portfolio Symbols =====")
        symbols = self.portfolio_manager.get_portfolio_symbols(self.current_portfolio_id)

        if not symbols:
            print("This portfolio doesn't have any symbols yet.")
            return

        #table headers with index num
        print(
            f"{'No.':<4} {'Ticker':<8} {'Sector':<23} {'Last Price':<12} {'Change':<10} {'Change %':<10} {'Shares':<10} {'Value':<12}")
        print("-" * 90)

        for index, symbol in enumerate(symbols, 1):  # Start index from 1
            metrics = self.portfolio_manager.calculate_symbol_metrics(symbol)

            # Format data for display
            price = f"${metrics['current_price']:.2f}" if metrics['current_price'] else "N/A"
            day_change = f"${metrics['day_change']:.2f}" if metrics['day_change'] else "N/A"
            day_change_pct = f"{metrics['day_change_percent']:.2f}%" if metrics['day_change_percent'] else "N/A"
            shares = f"{metrics['current_shares']:.2f}" if metrics['current_shares'] else "0.00"
            value = f"${metrics['current_value']:.2f}" if metrics['current_value'] else "$0.00"
            sector = symbol.sector if symbol.sector else "N/A"

            print(
                f"{index:<4} {symbol.ticker:<8} {sector:<23} {price:<12} {day_change:<10} {day_change_pct:<10} {shares:<10} {value:<12}")

        #select symbol number
        choice = input("\nSelect a symbol number to view transactions (0 to go back): ")
        if choice.isdigit():
            choice = int(choice)
            if choice == 0:
                return
            if 1 <= choice <= len(symbols):
                self.view_symbol_transactions(symbols[choice - 1])  # Adjust index for 0-based list

    #view transactions for a symbol
    def view_symbol_transactions(self, symbol: Symbol):
        print(f"\n===== Transactions for {symbol.ticker} =====")

        if not symbol.transactions:
            print("No transactions for this symbol yet.")
            return

        print(f"{'ID':<5} {'Type':<6} {'Shares':<10} {'Price':<12} {'Cost':<10} {'Date':<11}")
        print("-" * 80)

        for transaction in symbol.transactions:
            t_id = f"{transaction.transaction_id}"
            t_type = f"{transaction.transaction_type}"
            shares = f"{transaction.shares:.2f}"
            price = f"${transaction.price:.2f}"
            cost = f"${transaction.transaction_cost:.2f}"
            date = f"{transaction.transaction_date}"
            print(f"{t_id:<5} {t_type:<6} {shares:<10} {price:<12} {cost:<10} {date:<12}")

    #add transaction for symbol in current portfolio
    def add_transaction(self):
        print("\n===== Add Transaction =====")

        symbols = self.portfolio_manager.get_portfolio_symbols(self.current_portfolio_id)
        if not symbols:
            print("⚠️ No symbols in portfolio. Add stocks before recording a transaction.")
            return

        print("\nAvailable Symbols:")
        for i, symbol in enumerate(symbols, 1):
            print(f"{i}. {symbol.ticker} ({symbol.sector})")

        choice = input("\nSelect a symbol number (or type 'b' to go back): ")
        if choice.lower() == 'b':
            return
        if not choice.isdigit() or int(choice) < 1 or int(choice) > len(symbols):
            print("❌ Invalid selection.")
            return

        selected_symbol = symbols[int(choice) - 1]

        transaction_type = input("Enter transaction type (buy/sell): ").strip().lower()
        if transaction_type not in ["buy", "sell"]:
            print("❌ Invalid transaction type.")
            return

        try:
            shares = float(input("Enter number of shares: ").strip())
            if shares <= 0:
                raise ValueError
        except ValueError:
            print("❌ Invalid share quantity.")
            return

        #fetch latest price
        stock_data = self.portfolio_manager.yfinance_source.fetch_data(selected_symbol.ticker)
        if "error" in stock_data:
            print(f"⚠️ Error fetching stock price: {stock_data['error']}")
            return

        latest_price = stock_data["last_price"]
        print(f"💰 Latest price for {selected_symbol.ticker}: ${latest_price:.2f}")

        use_fetched_price = input("Use this price? (Y/N): ").strip().lower()
        if use_fetched_price == "n":
            try:
                manual_price = float(input("Enter your own price: ").strip())
            except ValueError:
                print("❌ Invalid input. Using fetched price.")
                manual_price = latest_price
        else:
            manual_price = latest_price

        transaction_cost = 0  # Modify if transaction fees exist
        transaction_date = datetime.datetime.now().strftime("%Y-%m-%d")

        #call `add_transaction()` from PortfolioManager
        transaction_id = self.portfolio_manager.add_transaction(
            selected_symbol.symbol_id, transaction_type, shares, manual_price, transaction_cost, transaction_date
        )

        if transaction_id != -1:
            print(
                f"✅ {transaction_type.capitalize()} of {shares} shares at ${manual_price:.2f} recorded successfully!")

    #summary of current portfolio
    def view_portfolio_summary(self):
        print("\n===== Portfolio Summary =====")
        metrics = self.portfolio_manager.calculate_portfolio_metrics(self.current_portfolio_id)

        #portfolio details
        portfolio_result = self.db_manager.execute_query(
            "SELECT name FROM portfolios WHERE portfolio_id = ?",
            (self.current_portfolio_id,)
        )

        if not portfolio_result:
            print("❌ Portfolio not found.")
            return

        portfolio_name = portfolio_result[0][0]

        print(f"Portfolio: {portfolio_name}")
        print(f"Total Investment: ${abs(metrics['total_investment']):.2f}" if metrics['total_investment'] >= 0 else f"Total Investment: -${abs(metrics['total_investment']):.2f}")
        print(f"Current Value: ${abs(metrics['total_current_value']):.2f}" if metrics['total_current_value'] >= 0 else f"Current Value: -${abs(metrics['total_current_value']):.2f}")
        print(f"Unrealised P/L: ${abs(metrics['total_unrealised_pl']):.2f}" if metrics['total_unrealised_pl'] >= 0 else f"Unrealised P/L: -${abs(metrics['total_unrealised_pl']):.2f} ({metrics['total_unrealised_pl_percent']:.2f}%)")
        print(f"Realised P/L: ${abs(metrics['total_realised_pl']):.2f}" if metrics['total_realised_pl'] >= 0 else f"Realised P/L: -${abs(metrics['total_realised_pl']):.2f}")
        print("\n===== Holdings =====")
        print(
            f"{'Ticker':<8} {'Sector':<23} {'Shares':<10} {'Avg Cost':<12} {'Current':<12} {'Value':<12} {'Unreal P/L':<12} {'Unreal P/L %':<12}")
        print("-" * 125)

        def format_currency(value):
            if value < 0:
                return f"-${abs(value):.2f}"
            else:
                return f"${value:.2f}"

        for symbol in metrics["symbols"]:
            ticker = symbol["ticker"]
            sector = symbol.get("sector", "N/A")
            shares = f"{symbol['current_shares']:.2f}"
            avg_cost = format_currency(symbol['avg_cost'])
            current = format_currency(symbol['current_price'])
            value = format_currency(symbol['current_value'])
            unreal_pl = format_currency(symbol['unrealised_pl'])
            unreal_pl_pct = f"{symbol['unrealised_pl_percent']:.2f}%"

            print(
                f"{ticker:<8} {sector:<23} {shares:<10} {avg_cost:<12} {current:<12} {value:<12} {unreal_pl:<12} {unreal_pl_pct:<12}")

    #analyze sector exposure
    def analyse_sector_exposure(self):
        """Analyzes sector exposure and suggests rebalancing if needed."""
        print("\n===== Sector Exposure Analysis =====")

        if not self.current_portfolio_id:
            print("❌ No portfolio selected.")
            return

        exposure = self.portfolio_manager.calculate_sector_exposure(self.current_portfolio_id, self.current_user_id)

        if not exposure:
            print("⚠️ No data available for analysis.")
            return

        # Fetch risk tolerance
        user_risk_tolerance = self.user_manager.get_user_risk_tolerance(self.current_user_id)
        risk_thresholds = {"Low": 20.0, "Medium": 30.0, "High": 40.0}
        default_threshold = risk_thresholds.get(user_risk_tolerance, 20.0)

        print(f"\n🔹 Your current risk tolerance: {user_risk_tolerance} (Threshold: {default_threshold}%)")

        total_exposure = sum(data["percentage"] for data in exposure.values())
        if abs(total_exposure - 100) > 0.01:
            print("⚠️ Warning: Portfolio exposure percentages do not sum to 100%. Adjust calculations if necessary.")

        # Retrieve all symbols in portfolio
        symbols = self.portfolio_manager.get_portfolio_symbols(self.current_portfolio_id)

        if len(symbols) <= 1:
            print("⚠️ Cannot rebalance. Your portfolio has only one stock.")
            print("📌 Consider adding more stocks to diversify.")
            return

        choice = input("\nWould you like to see rebalancing suggestions? (y/n): ").strip().lower()
        if choice != 'y':
            print("✅ Skipped rebalancing suggestion.")
            return

        risk_choice = input("Do you want to manually enter a threshold? (y/n): ").strip().lower()
        if risk_choice == 'y':
            threshold_input = input(f"Enter threshold percentage for overexposure [{default_threshold}]: ").strip()
            try:
                threshold = float(threshold_input) if threshold_input else default_threshold
                if threshold <= 0 or threshold > 100:
                    print(f"⚠️ Invalid range. Using default threshold of {default_threshold}%.")
                    threshold = default_threshold
            except ValueError:
                print(f"❌ Invalid input. Using default threshold of {default_threshold}%.")
                threshold = default_threshold
        else:
            threshold = default_threshold

        rebalancing_data = self.portfolio_manager.suggest_rebalancing(self.current_portfolio_id, self.current_user_id,
                                                                      threshold)

        over_exposed_sectors = rebalancing_data.get("over_exposed", {})
        under_exposed_sectors = rebalancing_data.get("under_exposed", {})

        tradable_stocks = any(
            self.portfolio_manager.calculate_symbol_metrics(s)["current_shares"] > 0 for s in symbols
        )
        if not tradable_stocks:
            print("⚠️ Rebalancing not possible. No tradable stocks available.")
            return

        if not over_exposed_sectors:
            print("✅ No rebalancing needed. Your portfolio is well-balanced.")
            return

        rebalance_choice = input("\nWould you like to proceed with automatic rebalancing? (y/n): ").strip().lower()
        if rebalance_choice == 'y':
            self.portfolio_manager.auto_rebalance(
                self.current_portfolio_id, self.current_user_id, over_exposed_sectors, under_exposed_sectors, threshold
            )
        else:
            print("✅ Rebalancing skipped for now.")

    #exporting portfolio data to CSV
    def export_portfolio(self):
        print("\n===== Export Portfolio =====")

        portfolio_result = self.db_manager.execute_query(
            "SELECT name FROM portfolios WHERE portfolio_id = ?",
            (self.current_portfolio_id,)
        )

        if not portfolio_result:
            print("❌ Portfolio not found.")
            return

        while True:
            directory = input("Enter the directory to save the report (or type 'b' to go back): ").strip()
            if directory.lower() == 'b':
                return


            file_path = self.portfolio_manager.export_portfolio_to_csv(
                self.current_portfolio_id, self.current_user_id, directory
            )
            if file_path:
                print(f"✅ Portfolio exported successfully to: {file_path}")
                return
            else:
                print("⚠️ Error exporting portfolio. Please try again or type 'b' to go back.")

    #run the application
    def run(self):
        create_database()  # this is to ensure database exists before running

        while True:
            self.display_menu()
            choice = input("\nEnter your choice: ").strip()

            if choice == "0":
                print("👋 Thank you for using Portfolio Manager. Goodbye!")
                break
            elif choice == "1":
                self.beginner_guide()

            # If the user is NOT logged in
            elif self.current_user_id is None:
                if choice == "2":
                    self.register()
                elif choice == "3":
                    self.login()
                else:
                    print("❌ Invalid choice. Please try again.")

            # If the user is logged in but has NOT selected a portfolio
            elif self.current_portfolio_id is None:
                if choice == "2":
                    self.create_portfolio()
                elif choice == "3":
                    self.view_portfolios()
                elif choice == "4":
                    self.user_manager.change_risk_tolerance(self.current_user_id)
                elif choice == "5":
                    self.logout()
                else:
                    print("❌ Invalid choice. Please try again.")

            # If the user is inside a selected portfolio
            else:
                if choice == "2":
                    self.add_symbol()
                elif choice == "3":
                    self.view_symbols()
                elif choice == "4":
                    self.add_transaction()
                elif choice == "5":
                    self.view_portfolio_summary()
                elif choice == "6":
                    self.analyse_sector_exposure()
                elif choice == "7":
                    self.export_portfolio()
                elif choice == "8":
                    self.current_portfolio_id = None
                else:
                    print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    ui = MainApp()
    ui.run()
