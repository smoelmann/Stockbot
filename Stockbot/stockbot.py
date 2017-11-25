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
        init_data   -   Gets a Pandas DataFrame with relevant information about the stock and saves it to a csv file with path from Stock.path
        init_data_csv
                    -   Gets a Pandas DataFrame from a csv file with the path from Stock.path
        update_data -   Appends new data to existing data. Also saves to local csv
        splot       -   [avg]: Array of averages that should be plotted. If averages have not been computed, do so and add them to data
        get_avg     -   interval: The number of days that should be taken average of including present day
                        Finds the average closing price over 'interval' number of days and adds it to daily_data.avg_price
        view_data   -   Prints the contents of Stock.data to the console
        rule_buy    -   Returns the latest index where the rx-avg is lower than ry-avg, and rx-avg is lower than daily closing price
        rule_sell   -   Returns the latest index where the rx-avg is higher than ry-avg, and rx-avg is higher than daily closing price

    """
    def __init__(self, symbol):
        """
        """
        self.symbol = symbol.upper()
        # self.name = self.get_name()
        self.path = "C:\\Stockbot\\Stocks\\{s}.csv".format(s=self.symbol)
        # self.data = self.init_data()
        self.data = self.init_data_csv()
        self.last_action = (0,0)  # Tuple of buy/sell and date

    def init_data(self):
        """
        """
        end = datetime.today()
        start = end - timedelta(days=1000)

        df = web.DataReader(self.symbol, "yahoo", start, end)
        df.to_csv(path_or_buf=self.path,columns=["High","Low","Close","Volume"])
        df = pd.read_csv(filepath_or_buffer=self.path)
        return df

    def init_data_csv(self):
        """
        """
        return pd.read_csv(self.path)

    def update_data(self):
        pass

    def splot(self,avg=None):
        """
        """
        if avg == None:
            self.data.plot(x="Date",y="Close",grid=True)
        else:
            self.data.plot(x=self.data.index,y=avg,grid=True,ylim=(max(self.data["Close"]*1.1),min(self.data["Close"])*0.9))
        plt.gca().invert_yaxis()
        plt.show()

    def get_avg(self,n):
        """
        """
        col = "avg_{n}".format(n=n)

        prices = self.data["Close"]
        dates = self.data["Date"]

        self.data[col] = self.data["Close"].copy()
        d = deque()
        for idx, price in enumerate(prices):
            if not np.isnan(price):
                if len(d) < n:
                    d.append(price)
                else:
                    d.popleft()
                    d.append(price)

                if len(d) == n:
                    avg = sum(d)/n
                    self.data.loc[idx, col] = avg
                else:
                    self.data.loc[idx, col] = np.nan
            else:
                self.data.loc[idx, col] = np.nan
        return self.data

    def view_data(self):
        """
        """
        print("{s}  {n}\n{p}\n{d}".format(n=self.name,s=self.symbol,p=self.path,d=self.data))

    def rule_buy(self, rx, ry):
        """
        """
        col_x = "avg_{x}".format(x=rx)
        col_y = "avg_{y}".format(y=ry)

        # Assert if columns exist. If not get them
        if not (col_x in self.data.columns):
            self.get_avg(rx)
        if not (col_y in self.data.columns):
            self.get_avg(ry)

        for idx in reversed(self.data.index):
            c = self.data.loc[idx, "Close"]
            x = self.data.loc[idx, col_x]
            y = self.data.loc[idx, col_y]
            if (c > y) and (x < y):
                print("Date: {d} (idx={i}), C: {c}, X: {x}, Y: {y}".format(d=self.data.loc[idx, "Date"], i=idx, c=c, x=x, y=y))
                # return idx

    def rule_sell(self, rx, ry):
        """
        """
        col_x = "avg_{x}".format(x=rx)
        col_y = "avg_{y}".format(y=ry)

        # Assert if columns exist. If not get them
        if not (col_x in self.data.columns):
            self.get_avg(rx)
        if not (col_y in self.data.columns):
            self.get_avg(ry)

        for idx in reversed(self.data.index):
            c = self.data.loc[idx, "Close"]
            x = self.data.loc[idx, col_x]
            y = self.data.loc[idx, col_y]
            if (c < x) and (x > y):
                # print("Date: {d} (idx={i}), C: {c}, X: {x}, Y: {y}".format(d=self.data.loc[idx, "Date"], i=idx, c=c, x=x, y=y))
                return idx

def simulate_market(stock, start_money, avg=(2,10)):
    """ avg  - the lowest and highest averages to be examined
    """
    start, end = avg

    # Get all averages from start to, and including, end intervals
    print("Calculating averages...")
    for x in range(start, end + 1):
        col_x = "avg_{x}".format(x=x)

        # Assert if Stock has y-avg. If not, get it
        if not (col_x in stock.data.columns):
            stock.get_avg(x)

    # Variables to contain max and min results
    max_money = 0
    max_avg = (0,0)
    min_money = np.inf
    min_avg = (0,0)
    min_num_purchases = 0
    max_num_purchases = 0
    max_trades = 0
    max_t = 0
    max_t_xy = (0,0)

    # Loop across averages and find the optimal intervals, only use y where y>x
    print("Calculating optimmal avgerages...")
    for x in range(start, end):
        col_x = "avg_{x}".format(x=x)
        gen = (y for y in range(start + 1, end + 1) if y > x)
        for y in gen:
        # for y in range(start + 1, end + 1):
            # Simulate buying and selling for x- and y-avg
            # Initializing variables
            money = start_money
            num_bought = 0
            num_purchases = 0
            mode = "buy"
            idx = max([x,y])
            idx_max = stock.data.last_valid_index()
            col_y = "avg_{y}".format(y=y)

            while idx <= idx_max:
                price = stock.data.loc[idx, "Close"]
                avg_x = stock.data.loc[idx, col_x]
                avg_y = stock.data.loc[idx, col_y]

                # Looking to buy
                if mode == "buy":
                    # Buy signal
                    if (price > y) and (x < y):
                        mode = "sell"
                        num_bought = money / price
                        money = 0
                        num_purchases +=1

                # Looking to sell
                if mode == "sell":
                    # Sell signal
                    if (price < x) and (x > y):
                        mode = "buy"
                        money = num_bought * price
                        num_bought = 0
                        num_purchases += 1
                # Increment idx
                idx += 1

            # Finally sell all to see profit
            money = num_bought * price

            # # Printing result of x-, y-avg
            # print("Avg: {x}  {y}\nGross: {profit}  ({diff})".format(x=x, y=y, profit=money/start_money, diff=money-start_money))
            # Logging max and min values
            if money >= max_money and num_purchases > 1:
                max_money = money
                max_avg = (x, y)
                max_num_purchases = num_purchases
            if money <= min_money and num_purchases > 1:
                min_money = money
                min_avg = (x, y)
                min_num_purchases = num_purchases
            if num_purchases >= max_trades:
                max_trades = num_purchases
                max_t = money
                max_t_xy = (x, y)

    maxx, maxy = max_avg
    minx, miny = min_avg
    tx, ty = max_t_xy
    print("MAX:: {p}%  ({x}, {y}). Num {n}".format(p=max_money/start_money*100-100, x=maxx, y=maxy, n=max_num_purchases))
    print("MIN:: {p}%  ({x}, {y}). Num {n}".format(p=min_money/start_money*100-100, x=minx, y=miny, n=min_num_purchases))
    print("TRD:: {p}%  ({x}, {y}). Num {n}".format(p=max_t/start_money*100-100, x=tx, y=ty, n=max_trades))
# End simulate_market method


if __name__ == "__main__":
    print("Getting data...")
    test_stock = Stock("AMZN")
    test_stock.get_avg(33)
    test_stock.get_avg(34)
    # test_stock.view_data()
    # test_stock.splot()
    # test_stock.rule_buy(33, 34)
    # test_stock.rule_sell(15, 50)
    # test_stock.splot(["Close","avg_16", "avg_17"])

    simulate_market(test_stock, 10000, (2,50))
    test_stock.splot(["Close","avg_33", "avg_34"])









# Whitespace
