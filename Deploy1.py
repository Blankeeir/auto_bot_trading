#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import tpqoa
from datetime import datetime, timedelta
import time

class Real_Test_711(tpqoa.tpqoa):
    def __init__(self, conf_file, instrument, bar_length, window, units):
        super().__init__(conf_file)
        self.instrument = instrument
        self.bar_length = pd.to_timedelta(bar_length)
        self.tick_data = pd.DataFrame()
        self.raw_data = None
        self.data = None 
        self.last_bar = None
        self.units = units
        self.position = 0
        self.profits = [] 
        
        # Strategy-specific attributes
        self.window = window
        self.tc = 0.000059  # transaction cost
        self.ema_s = 5
        self.ema_l = 20
        self.rsi_s = 7
        self.rsi_l = 14
        self.rsi_l_threshold = 50
        self.rsi_s_threshold = 50
        self.donchian_length = 5
        self.atr_length = 14
        self.atr_multiplier = 1.0
        self.take_profit_multiplier = 2.0
    
    def get_most_recent(self, days = 5):
        while True:
            time.sleep(2)
            now = datetime.utcnow()
            now = now - timedelta(microseconds = now.microsecond)
            past = now - timedelta(days = days)
            df = self.get_history(instrument = self.instrument, start = past, end = now,
                                   granularity = "S5", price = "M", localize = False).c.dropna().to_frame()
            df.rename(columns = {"c":self.instrument}, inplace = True)
            df = df.resample(self.bar_length, label = "right").last().dropna().iloc[:-1]
            self.raw_data = df.copy()
            self.last_bar = self.raw_data.index[-1]
            if pd.to_datetime(datetime.utcnow()).tz_localize("UTC") - self.last_bar < self.bar_length:
                break
            
    def start_trading(self, days, max_attempts = 5, wait = 20, wait_increase = 0): 
        attempt = 0
        success = False
        while True:
            try:
                self.get_most_recent(days)
                self.stream_data(self.instrument)
            except Exception as e:
                print(e, end = " | ")
            else:
                success = True
                break    
            finally:
                attempt +=1
                print("Attempt: {}".format(attempt), end = '\n')
                if success == False:
                    if attempt >= max_attempts:
                        print("max_attempts reached!")
                        try: 
                            time.sleep(wait)
                            self.terminate_session(cause = "Unexpected Session Stop (too many errors).")
                        except Exception as e:
                            print(e, end = " | ")
                            print("Could not terminate session properly!")
                        finally: 
                            break
                    else: 
                        time.sleep(wait)
                        wait += wait_increase
                        self.tick_data = pd.DataFrame()
        
    def on_success(self, time, bid, ask):
        print(self.ticks, end = '\r', flush = True)
        
        recent_tick = pd.to_datetime(time)
        
        
        # define stop
        if recent_tick.time() >= pd.to_datetime("11:00").time():
            self.stop_stream = True
        
        df = pd.DataFrame({self.instrument:(ask + bid)/2}, 
                          index = [recent_tick])
        self.tick_data = pd.concat([self.tick_data, df]) 
        
        if recent_tick - self.last_bar >= self.bar_length:
            self.resample_and_join()
            self.define_strategy()
            self.execute_trades()
            
    def resample_and_join(self):
        self.raw_data = pd.concat([self.raw_data, self.tick_data.resample(self.bar_length, 
                                                                          label="right").last().ffill().iloc[:-1]])
        self.tick_data = self.tick_data.iloc[-1:]
        self.last_bar = self.raw_data.index[-1]
        
    def define_strategy(self):
        df = self.raw_data.copy()
        
        df["returns"] = np.log(df[self.instrument] / df[self.instrument].shift(1))
        df["EMA_S"] = df[self.instrument].ewm(span=self.ema_s, min_periods=self.ema_s).mean()
        df["EMA_L"] = df[self.instrument].ewm(span=self.ema_l, min_periods=self.ema_l).mean()

        df["U"] = np.where(df[self.instrument].diff() > 0, df[self.instrument].diff(), 0)
        df["D"] = np.where(df[self.instrument].diff() < 0, -df[self.instrument].diff(), 0)
        df["MA_U"] = df.U.rolling(self.rsi_s).mean()
        df["MA_D"] = df.D.rolling(self.rsi_s).mean()
        df["MA_U_l"] = df.U.rolling(self.rsi_l).mean()
        df["MA_D_l"] = df.D.rolling(self.rsi_l).mean()
        df["RSI_S"] = df.MA_U / (df.MA_U + df.MA_D) * 100
        df["RSI_L"] = df.MA_U_l / (df.MA_U_l + df.MA_D_l) * 100

        df['Donchian_Upper'] = df[self.instrument].rolling(window=self.donchian_length).max()
        df['Donchian_Lower'] = df[self.instrument].rolling(window=self.donchian_length).min()
        df['Donchian_Mid'] = (df['Donchian_Upper'] + df['Donchian_Lower']) / 2

        df['ATR'] = df[self.instrument].rolling(window=self.atr_length).max() - df[self.instrument].rolling(window=self.atr_length).min()

        df['stop_loss_long'] = df[self.instrument] - df['ATR'] * self.atr_multiplier
        df['take_profit_long'] = df[self.instrument] + df['ATR'] * self.take_profit_multiplier
        df['stop_loss_short'] = df[self.instrument] + df['ATR'] * self.atr_multiplier
        df['take_profit_short'] = df[self.instrument] - df['ATR'] * self.take_profit_multiplier

        df["position"] = np.where((df["EMA_S"] > df["EMA_L"]) &
                                  (df['RSI_S'] > self.rsi_l_threshold) &
                                  (df['RSI_L'] > self.rsi_l_threshold) &
                                  (df[self.instrument] > df['Donchian_Mid']),
                                  1, np.nan)

        df["position"] = np.where((df["EMA_S"] < df["EMA_L"]) &
                                  (df['RSI_S'] < self.rsi_s_threshold) &
                                  (df['RSI_L'] < self.rsi_s_threshold) &
                                  (df[self.instrument] < df['Donchian_Mid']),
                                  -1, df['position'])

        df["position"] = np.where((df["position"] == 1) &
                                  ((df[self.instrument] <= df['stop_loss_long']) |
                                   (df[self.instrument] >= df['take_profit_long'])),
                                  0, df["position"])

        df["position"] = np.where((df["position"] == -1) &
                                  ((df[self.instrument] >= df['stop_loss_short']) |
                                   (df[self.instrument] <= df['take_profit_short'])),
                                  0, df["position"])

        self.data = df.copy()
        
    def execute_trades(self):
        if self.data["position"].iloc[-1] == 1:
            if self.position == 0:
                order = self.create_order(self.instrument, self.units, suppress = True, ret = True)
                self.report_trade(order, "GOING LONG")  
            elif self.position == -1:
                order = self.create_order(self.instrument, self.units * 2, suppress = True, ret = True) 
                self.report_trade(order, "GOING LONG")  
            self.position = 1
        elif self.data["position"].iloc[-1] == -1: 
            if self.position == 0:
                order = self.create_order(self.instrument, -self.units, suppress = True, ret = True)
                self.report_trade(order, "GOING SHORT")  
            elif self.position == 1:
                order = self.create_order(self.instrument, -self.units * 2, suppress = True, ret = True)
                self.report_trade(order, "GOING SHORT")  
            self.position = -1
        elif self.data["position"].iloc[-1] == 0: 
            if self.position == -1:
                order = self.create_order(self.instrument, self.units, suppress = True, ret = True) 
                self.report_trade(order, "GOING NEUTRAL")  
            elif self.position == 1:
                order = self.create_order(self.instrument, -self.units, suppress = True, ret = True)
                self.report_trade(order, "GOING NEUTRAL")  
            self.position = 0
    
    def report_trade(self, order, going):  
        time = order["time"]
        units = order["units"]
        price = order["price"]
        pl = float(order["pl"])
        self.profits.append(pl)
        cumpl = sum(self.profits)
        print("\n" + 100* "-")
        print("{} | {}".format(time, going))
        print("{} | units = {} | price = {} | P&L = {} | Cum P&L = {}".format(time, units, price, pl, cumpl))
        print(100 * "-" + "\n")  
        
    def terminate_session(self, cause):
        self.stop_stream = True
        if self.position != 0:
            close_order = self.create_order(self.instrument, units = -self.position * self.units,
                                            suppress = True, ret = True) 
            self.report_trade(close_order, "GOING NEUTRAL")
            self.position = 0
        print(cause, end = " | ")

if __name__ == "__main__":
        
    trader = Real_Test_711(r"C:\Users\Chia W Y\Downloads\Algo\Part4_Materials\Oanda\oanda.cfg", "EUR_USD", "1min", window = 1, units = 100000)
    trader.start_trading(days = 5, max_attempts =  5, wait = 20, wait_increase = 0)


# In[ ]:




