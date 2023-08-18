import pandas as pd
import numpy as np
import tpqoa
from datetime import datetime, timedelta, timezone
import time

class Trader(tpqoa.tpqoa):
    def __init__(self, conf_file, instrument, bar_length, units, duration):
        super().__init__(conf_file)
        self.instrument = instrument
        self.bar_length = pd.to_timedelta(bar_length)
        self.tick_data = pd.DataFrame()
        self.start = datetime.utcnow()
        self.end = self.start + timedelta(minutes=duration)
        self.raw_data = None
        self.data = None 
        self.last_bar = None
        self.units = units
        self.position = 0
        self.profits = []
        self.duration = duration

        self.start_trade_session()
    
    # Begins trading session with error catching
    def start_trade_session(self, days=5, max_attempts=None, sleep_period=15, sleep_increase=0):
        attempt = 0
        success = False
        while True:
            try:
                self.get_most_recent(days)
                self.stream_data(self.instrument)
            except Exception as e:
                print('Error:', e)
            else:
                success = True
                break
            finally:
                attempt += 1
                print('Attempt: {}'.format(attempt), end='\n')
                if not success:
                    if max_attempts is not None and attempt >= max_attempts:
                        print('MAX ATTEMPTS REACHED')
                        try:
                            time.sleep(sleep_period)
                            self.terminate_session(
                                cause='TOO MANY ERRORS - SESSION TERMINATED.'
                            )
                        except Exception as e:
                            print('Error:', e)
                            print('COULD NOT TERMINATE SESSION.')
                        finally:
                            break
                    else: # TRY AGAIN
                        time.sleep(sleep_period)
                        sleep_period += sleep_increase
                        self.tick_data = pd.DataFrame()

    # Get recent data, with specified time interval
    def get_most_recent(self, days=5):
        time.sleep(1)
        print('-' * 50)
        print('ATTEMPTING TO MERGE...')
        print('REQUIRE UNDER {} SECONDS'.format(self.bar_length.seconds))
        now = datetime.utcnow()
        now = now - timedelta(microseconds=now.microsecond)
        past = now - timedelta(days=days)
        df = (self.get_history(
                instrument = self.instrument,
                start = past, 
                end = now,
                granularity = 'S5',
                price = 'M',
                localize = True
            )
            .c.dropna()
            .to_frame()
        )
        df.rename(columns = {'c':self.instrument}, inplace=True)
        low = df.resample(self.bar_length, label='right').min().dropna()
        high = df.resample(self.bar_length, label='right').max().dropna()
        df = df.resample(self.bar_length, label='right').last().dropna().iloc[:-1]
        
        self.raw_data = df.copy()
        self.raw_data['High'] = high
        self.raw_data['Low'] = low
        # print(self.raw_data)
        self.last_bar = self.raw_data.index[-1]
        
        print('Seconds: {}'.format((datetime.utcnow() - self.last_bar).seconds))

        if datetime.utcnow() - self.last_bar >= self.bar_length:
            print('-----VERIFY THAT BOT IS RUNNING DURING TRADING HOURS-----')
            self.get_most_recent()
        else:
            print('SUCCESSFULLY MERGED')
            print('-' * 50)

    def close_position(self):
        if self.position == 1:
            order = self.create_order(
                self.instrument,
                -self.units,
                suppress = True,
                ret = True
            )
            self.report_trade(order, 'GOING NEUTRAL')
        elif self.position == -1:
            order = self.create_order(
                self.instrument,
                self.units,
                suppress = True,
                ret = True
            )
            self.report_trade(order, 'GOING NEUTRAL')
        print('\nSESSION OVER')
        self.position = 0

    # End the trading session
    def end_trade_session(self):
        self.stop_stream = True

        if self.position != 0:
            close_order = self.create_order(
                self.instrument,
                units = -self.position * self.units,
                suppress = True,
                ret = True
            )
            self.report_trade(close_order, "GOING NEUTRAL")
            self.position = 0
        print('\nENDING TRADING SESSION...')

    def on_success(self, time, bid, ask):
        recent_tick = pd.to_datetime(time).replace(tzinfo=None)
        print(self.ticks, end='\r', flush=True)
        if recent_tick >= self.end:
            self.end_trade_session()
            return

        df = pd.DataFrame({self.instrument:(ask + bid) / 2}, index=[recent_tick])
        self.tick_data = pd.concat([self.tick_data, df])
        
        if self.ticks == 1 or recent_tick - self.last_bar > self.bar_length:
            self.resample_and_join()
            self.define_strategy()
            self.execute_trades()
    
    def resample_and_join(self):
        temp = self.tick_data.resample(self.bar_length, label="right").last().ffill().iloc[:-1]
        temp['High'] = self.tick_data.resample(self.bar_length, label='right').max().dropna()
        temp['Low'] = self.tick_data.resample(self.bar_length, label='right').min().dropna()
        self.raw_data = pd.concat([
            self.raw_data, 
            temp
        ])
        self.tick_data = self.tick_data.iloc[-1:]
        # print(self.raw_data)
        self.last_bar = self.raw_data.index[-1]
    
    def define_strategy(self):
        df = self.raw_data.copy()
        df['position'] = 0
        self.data = df.copy()
        pass
    
    def execute_trades(self):
        if self.data['position'].iloc[-1] == 1:
            if self.position == 0:
                order = self.create_order(self.instrument, self.units, suppress=True, ret=True)
                self.report_trade(order, 'GOING LONG')
            elif self.position == -1:
                order = self.create_order(self.instrument, self.units * 2, suppress=True, ret=True) 
                self.report_trade(order, 'GOING LONG')
            else:
                print('STAYING LONG')
            self.position = 1
        elif self.data['position'].iloc[-1] == -1: 
            if self.position == 0:
                order = self.create_order(self.instrument, -self.units, suppress=True, ret=True)
                self.report_trade(order, 'GOING SHORT')
            elif self.position == 1:
                order = self.create_order(self.instrument, -self.units * 2, suppress=True, ret=True)
                self.report_trade(order, 'GOING SHORT')
            else:
                print('STAYING SHORT')
            self.position = -1
        elif self.data['position'].iloc[-1] == 0:
            if self.position == -1:
                order = self.create_order(self.instrument, self.units, suppress=True, ret=True) 
                self.report_trade(order, 'GOING NEUTRAL')
            elif self.position == 1:
                order = self.create_order(self.instrument, -self.units, suppress=True, ret=True) 
                self.report_trade(order, 'GOING NEUTRAL')
            else:
                print('STAYING NEUTRAL')
            self.position = 0
    
    # Print out trade statistics
    def report_trade(self, order, going):
        time = order['time']
        units = order['units']
        price = order['price']
        pl = float(order['pl'])
        self.profits.append(pl)
        cumpl = sum(self.profits)
        print('\n' + 50* '-')
        print('{} | {}'.format(time, going))
        print('{} | Units = {} | Price = {} | P&L = {} | Cumulative P&L = {}'.format(time, units, price, pl, cumpl))
        print(50 * '-' + '\n')  
    