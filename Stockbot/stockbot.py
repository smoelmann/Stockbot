import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data as web
from datetime import datetime, timedelta
from yahoo_finance import Share
from math import ceil, floor
from collections import deque

class Stock():
    """ Historical data of a Stock
    Attributes:
        symbol      -   The official name of the stock
        path        -   A path to the csv file containing information
        data        -   Pandas DataFrame with all daily data
        self.last_action
                    -   A tuple of the latest action (buy or sell) and the date

    Methods:
        init_data   -   Gets a Pandas DataFrame with relevant information about the stock and saves it to a csv file with path from Stock.path.
        init_data_csv
                    -   Gets a Pandas DataFrame from a csv file with the path from Stock.path.
        update_data -   *TODO* Appends new data to existing data. Also saves to local csv.
        splot       -   Plots a graph of closing price and closing averages specified in 'avg'.
        get_avg     -   Finds the average closing price over 'avg_interval' number of days and adds a column to Stock.data.
        print_data  -   Prints the Stock.data to the console.
        create_avg  -   Creates the
        do_rule_buy -   Asserts if a buy-signal should be triggered.
        rule_buy    -   Returns the latest index where Stock.do_rule_buy() returns True.
        do_rule_sell-   Asserts if a sell-signal should be triggered.
        rule_sell   -   Returns the latest index where Stock.do_rule_sell() returns True.

    """
    def __init__(self, symbol, path="C:\\Stockbot\\Stocks", num_days=1000):
        """
        params:
            symbol   - (String) The unique character combination indicating a certain share.
            path     - (String) Default "C:\\Stockbot\\Stocks". The path directory where the Stocks related csv will be stored.
            num_days - (Int) Default 1000. The number of days for data gathering including closing days.
        returns:
            None
        Initializing method.
        """
        self.symbol = symbol.upper()
        self.path = "C:\\Stockbot\\Stocks\\{s}.csv".format(s=self.symbol)
        # self.data = self.init_data(num_days)
        self.data = self.init_data_csv()
        self.last_action = (0,0)  # Tuple of buy/sell and date

    def init_data(self, num_days=1000):
        """
        params:
            num_days - (Int) Default 1000. Number of days to fetch data for, including closing days
        returns:
            (pandas.DataFrame) A DataFrame for the last num_days days' worth of stock data. Values [ High, Low, Close, Volume ] are kept.
        Fetches data from Yahoo Finance using pandas_datareader the last num_days days. Writes the resulting csv to path as {symbol}.csv which is subsecuently is read and returned.
        """
        end = datetime.today()
        start = end - timedelta(days=num_days)

        df = web.DataReader(self.symbol, "yahoo", start, end)
        df.to_csv(path_or_buf=self.path,columns=["High","Low","Close","Volume"])
        df = pd.read_csv(filepath_or_buffer=self.path)
        return df

    def init_data_csv(self):
        """
        params:
            None
        returns:
            (pandas.DataFrame) A DataFrame read from the csv stored in Stock.path.
        Fetches data from a csv stored in Stock.path.
        """
        return pd.read_csv(self.path)

    def update_data(self):
        """
        *TODO* Appends new data to existing data. Also saves to local csv.
        """
        pass

    def splot(self,avg=None):
        """
        params:
            avg - (List of Ints) Defualt None. If unchanged, plot only closing prices. Plot averages specified in avg.
        returns:
            None.
        Plots a graph of closing price and closing averages specified in 'avg'.
        """
        avgs = ["Close"]
        for avg_interval in avg:
            self.create_avg(avg_interval)
            avgs.append("avg_{avg_interval}".format(avg_interval=avg_interval))
        self.data.plot(x=self.data.index, y=avgs, grid=True, ylim=(max(self.data["Close"]*1.1),min(self.data["Close"])*0.9))
        plt.gca().invert_yaxis()
        plt.show()

    def print_data(self):
        """
        params:
            None.
        returns:
            None.
        Prints the Stock.data to the console.
        """
        print("{s}\n{p}\n{d}".format(s=self.symbol,p=self.path,d=self.data))

    def get_avg(self,avg_interval):
        """
        params:
            avg_interval - (Int) The interval of days that should be averaged.
        returns:
            (pandas.DataFrame) Stock.data including the newly created average column.
        Finds the average closing price over 'avg_interval' number of days and adds a column to Stock.data.
        """
        col = "avg_{avg_interval}".format(avg_interval=avg_interval)

        prices = self.data["Close"]
        dates = self.data["Date"]

        self.data[col] = self.data["Close"].copy()
        d = deque()
        for idx, price in enumerate(prices):
            if not np.isnan(price):
                if len(d) < avg_interval:
                    d.append(price)
                else:
                    d.popleft()
                    d.append(price)

                if len(d) == avg_interval:
                    avg = sum(d)/avg_interval
                    self.data.loc[idx, col] = avg
                else:
                    self.data.loc[idx, col] = np.nan
            else:
                self.data.loc[idx, col] = np.nan
        return self.data

    def create_avg(self, avg_interval):
        """
        params:
            avg_interval - (Int) The interval of days that should be averaged.
        returns:
            (pandas.DataFrame) Stock.data including the newly created average column, if any.
        Finds the average closing price over 'avg_interval' number of days and adds a column to Stock.data if the column does not already exsists.
        """
        if not (avg_interval in self.data.columns):
            df = self.get_avg(avg_interval)
        return df

    def do_rule_buy(self, idx, col_x, col_y):
        """
        params:
            idx   - (Int) The index of Stock.data that should be examined.
            col_x - (String) Name of the first column for comparison.
            col_y - (String) Name of the second column for comparison.
        returns:
            (Boolean) The evaluation of whether or not it would be recommended to buy this Stock based on the following rule: (closing_price > val_x and val_x < val_y).
        Asserts if a buy-signal should be triggered.
        """
        price = self.data.loc[idx, "Close"]
        avg_x = self.data.loc[idx, col_x]
        avg_y = self.data.loc[idx, col_y]

        if price > avg_x and avg_x < avg_y:
            return True
        else:
            return False

    def rule_buy(self, x, y):
        """
        params:
            x - (Int) The first average to be compared.
            y - (Int) The second average to be compared.
        returns:
            (Int) The latest index where a buy signal was triggered.
        Returns the latest index where Stock.do_rule_buy() returns True.
        """
        col_x = "avg_{x}".format(x=x)
        self.create_avg(x)

        col_y = "avg_{y}".format(y=y)
        self.create_avg(y)

        for idx in reversed(self.data.index):
            if self.do_rule_buy(idx, col_x, col_y):
                return idx

    def do_rule_sell(self, idx, col_x, col_y):
        """
        params:
            idx   - (Int) The index of Stock.data that should be examined.
            col_x - (String) Name of the first column for comparison.
            col_y - (String) Name of the second column for comparison.
        returns:
            (Boolean) The evaluation of whether or not it would be recommended to sell this Stock based on the following rule: (closing_price < val_x and val_x > val_y).
        Asserts if a sell-signal should be triggered.
        """
        price = self.data.loc[idx, "Close"]
        avg_x = self.data.loc[idx, col_x]
        avg_y = self.data.loc[idx, col_y]

        if price < avg_x and avg_x > avg_y:
            return True
        else:
            return False

    def rule_sell(self, x, y):
        """
        params:
            x - (Int) The first average to be compared.
            y - (Int) The second average to be compared.
        returns:
            (Int) The latest index where a sell signal was triggered.
        Returns the latest index where Stock.do_rule_sell() returns True.
        """
        col_x = "avg_{x}".format(x=x)
        self.create_avg(x)

        col_y = "avg_{y}".format(y=y)
        self.create_avg(y)

        for idx in reversed(self.data.index):
            if self.do_rule_sell(idx, col_x, col_y):
                return idx

def simulate_market(stock, start_money, avg=(2,10)):
    """ avg  - the lowest and highest averages to be examined
    """
    # Create all averages from start through end intervals
    start, end = avg
    for x in range(start, end + 1):
        col_x = "avg_{x}".format(x=x)
        stock.create_avg(x)

    # Variables to contain logging results
    max_money = 0
    max_avg = (0,0)
    max_num_purchases = 0

    # Loop across averages and find the optimal intervals, only use y where y > x + 1
    for x in range(start, end):
        col_x = "avg_{x}".format(x=x)
        gen = (y for y in range(start + 1, end + 1) if y > x + 1)
        for y in gen:
            # Initializing variables
            money, num_bought, num_purchases, mode = start_money, 0, 0, "buy"
            idx, idx_max = y, stock.data.last_valid_index()
            col_y = "avg_{y}".format(y=y)

            for idx in range(0, idx_max + 1):
                # Want to buy
                if mode == "buy" and stock.do_rule_buy(idx, col_x, col_y):
                    mode = "sell"
                    price = stock.data.loc[idx, "Close"]
                    num_bought, money = money / price, 0
                    num_purchases += 1

                # Want to sell
                if mode == "sell" and stock.do_rule_sell(idx, col_x, col_y):
                    mode = "buy"
                    price = stock.data.loc[idx, "Close"]
                    money, num_bought = num_bought * price, 0
                    num_purchases += 1

            # Finally sell all to see profit
            money = num_bought * price

            # # Printing result of x-, y-avg
            # print("Avg: {x}  {y}  {t}\nGross: {profit}  ({diff})\n\n\n".format(x=x, y=y, t=num_purchases, profit=round(money/start_money,3), diff=round(money-start_money,3)))

            # Logging max values
            if money >= max_money and num_purchases > 1:
                max_money = money
                max_avg = (x, y)
                max_num_purchases = num_purchases

    # Print logs
    maxx, maxy = max_avg
    print("MAX:: {p}%  ({x}, {y}). Num {n}".format(p=round(max_money/start_money*100,3), x=maxx, y=maxy, n=max_num_purchases))


if __name__ == "__main__":
    test_stock = Stock("AMZN")
    # test_stock.get_avg(2)
    # test_stock.print_data()
    # test_stock.rule_buy(3, 4)
    # test_stock.rule_sell(5, 6)
    # simulate_market(test_stock, 10000, (7,10))
    # test_stock.splot([11, 12])


"""
TODO:
Retry fetching data from web
Write the Stock.update_data() method
Create a proper test method
Check Stock.init_csv() in case no csv in Stock.path
Create notification system that provides insigh whether or not it recommends to buy/sell
"""
