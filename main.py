import time
import pandas as pd
import pandas_ta as ta
from binance.client import Client
from datetime import datetime, timedelta
from telegram import Bot
import csv
import os

class BreakoutScanner:
    def __init__(self):
        # Hardcoded credentials
        self.bot = Bot(token="7532851212:AAHOVx1esNWrtk2SJbBQHCXMae7Y-dKJR5o")
        self.client = Client("jU1j7eGK0QJayDX3WwaVcFHOuJfUO3MFU5J9ppfT6DiUslaAkwJrC6mf9lCXBVyU",
                             "aTQtuh5bFKsGxC3bJOPbL7IV4KC1DsQowxgbrO6awi4YbnfCp3BRU55SA58jhRXy")
        self.chat_id = "7771111812"

        self.symbols = [
    "PEPEUSDT",     "HUMAUSDT",     "SUIUSDT",     "WIFUSDT",     "UNIUSDT",
    "ENAUSDT",     "ANIMEUSDT",     "AAVEUSDT",     "VIRTUALUSDT",     "RVNUSDT",
    "NEIROUSDT",     "UMAUSDT",     "WLDUSDT",     "TRUMPUSDT",     "TAOUSDT",
    "AXLUSDT",     "KAIAUSDT",     "PNUTUSDT",     "ETHFIUSDT",     "ICPUSDT",
    "FETUSDT",     "TONUSDT",     "TRBUSDT",     "HMSTRUSDT",     "LPTUSDT",
    "PENGUUSDT",     "COMPUSDT",     "CRVUSDT",     "EIGENUSDT",     "BONKUSDT",
    "OPUSDT",     "SUSDT",     "NEARUSDT",     "ARBUSDT",     "COOKIEUSDT",
    "FLOKIUSDT",     "AIXBTUSDT",     "CAKEUSDT",     "HBARUSDT",     "TIAUSDT",
    "RAYUSDT",     "DEXEUSDT",     "APTUSDT",     "PENDLEUSDT",     "LDOUSDT",
    "MKRUSDT",     "NXPCUSDT",     "RENDERUSDT",     "MUBARAKUSDT",     "KAITOUSDT",
    "XLMUSDT",     "INITUSDT",     "WBTCUSDT",     "WCTUSDT",     "INJUSDT",
    "SEIUSDT",     "RUNEUSDT",     "ONDOUSDT",     "GALAUSDT",     "FILUSDT",
    "AUSDT",     "ETCUSDT",     "SOPHUSDT",     "ORDIUSDT",     "ZROUSDT",
    "ARKMUSDT",     "TURBOUSDT",     "ZENUSDT",     "CETUSUSDT",     "TUTUSDT",
    "GUNUSDT",     "VETUSDT",     "GPSUSDT",     "APEUSDT",     "PEOPLEUSDT",
    "ENSUSDT",     "OMUSDT",     "DYDXUSDT",     "MOVEUSDT",     "BOMEUSDT",
    "SHELLUSDT",     "WUSDT",     "1000SATSUSDT",     "BERAUSDT",     "ATOMUSDT",
    "LQTYUSDT",     "ACMUSDT",     "ALGOUSDT",     "SANDUSDT",     "API3USDT",
    "VANAUSDT",     "KDAUSDT",     "POLUSDT",     "SSVUSDT",     "ARUSDT",
    "RPLUSDT",     "SAGAUSDT",     "IOSTUSDT",     "ICXUSDT",     "GRTUSDT",
    "QNTUSDT",     "AXSUSDT",     "NEOUSDT",     "CHESSUSDT",     "JASMYUSDT",
    "BELUSDT",     "RADUSDT",     "IOUSDT",     "STXUSDT",     "IMXUSDT",
    "STRKUSDT",     "SXTUSDT",     "CFXUSDT",     "MEMEUSDT",     "ACTUSDT",
    "GMTUSDT",     "SUSHIUSDT",     "ZKUSDT",     "VOXELUSDT",     "NOTUSDT",
    "USUALUSDT",     "THETAUSDT",     "TSTUSDT",     "BIOUSDT",     "GMXUSDT",
    "LISTAUSDT",     "RSRUSDT",     "NILUSDT",     "THEUSDT",     "MOVRUSDT",
    "DEGOUSDT",     "TUSDT",     "CVCUSDT",     "LUNCUSDT",     "WBETHUSDT",
    "ACHUSDT",     "FUNUSDT",     "PYTHUSDT",     "MANTAUSDT",     "EGLDUSDT",
    "ZRXUSDT",     "LAYERUSDT",     "BANANAUSDT",     "HFTUSDT",     "JTOUSDT",
    "1MBABYDOGEUSDT",     "COWUSDT",     "MINAUSDT",     "XAIUSDT",     "DOGSUSDT",
    "ORCAUSDT",     "ROSEUSDT",     "PIXELUSDT",     "MEUSDT",     "METISUSDT",
    "BEAMXUSDT",     "FIDAUSDT",     "SLPUSDT",     "JOEUSDT",     "CGPTUSDT",
    "KAVAUSDT",     "RAREUSDT",     "HAEDALUSDT",     "HIGHUSDT",     "CHZUSDT",
    "LRCUSDT",     "SCRUSDT",     "REZUSDT",     "VANRYUSDT",     "AIUSDT",
    "ZECUSDT",     "BROCCOLI714USDT",     "KERNELUSDT",     "XTZUSDT",     "PHAUSDT",
    "MANAUSDT",     "IOTAUSDT",     "SOLVUSDT",     "LUNAUSDT",     "CELOUSDT",
    "SIGNUSDT",     "1000CATUSDT",     "RONINUSDT",     "SLFUSDT",     "SNXUSDT",
    "AEVOUSDT",     "PARTIUSDT",     "BNSOLUSDT",     "IOTXUSDT",     "PUNDIXUSDT",
    "XVGUSDT",     "BBUSDT",     "GLMRUSDT",     "1INCHUSDT",     "AWEUSDT",
    "BABYUSDT",     "AMPUSDT",     "HYPERUSDT",     "CHRUSDT",     "LUMIAUSDT",
    "BLURUSDT",     "BICOUSDT",     "CVXUSDT",     "STOUSDT",     "QTUMUSDT",
    "ONGUSDT",     "CKBUSDT",     "DASHUSDT",     "USTCUSDT",     "YFIUSDT",
    "ONEUSDT",     "ALTUSDT",     "BSWUSDT",     "COTIUSDT",     "ACXUSDT",
    "CYBERUSDT",     "FLOWUSDT",     "OGUSDT",     "ATMUSDT",     "FTTUSDT",
    "AGLDUSDT",     "ENJUSDT",     "AUCTIONUSDT",     "OMNIUSDT",     "STORJUSDT",
    "HOTUSDT",     "OGNUSDT",     "YGGUSDT",     "PORTALUSDT",     "ZILUSDT",
    "BAKEUSDT",     "ALPHAUSDT",     "RLCUSDT",     "EDUUSDT",     "PYRUSDT",
    "PHBUSDT",     "OSMOUSDT",     "C98USDT", ]

        self.timeframes = {
            "15m": Client.KLINE_INTERVAL_15MINUTE,
            "1h": Client.KLINE_INTERVAL_1HOUR,
            "4h": Client.KLINE_INTERVAL_4HOUR
        }

        self.settings = {
            "lookback": 200,
            "min_volume_ratio": 2.5,
            "price_change_threshold": 0.04,
            "rsi_range": (55, 75),
            "atr_multiplier": 1.8,
            "min_candle_body": 0.025,
            "capital": 3,
            "leverage": 20,
            "target_profit": 2
        }

        self.csv_file = "signals_log.csv"
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Symbol", "Timeframe", "Direction", "Entry", "TP1", "SL", "Confidence", "Profit", "Predicted Exit"])

    def fetch_market_data(self, symbol, interval):
        try:
            klines = self.client.get_klines(symbol=symbol, interval=interval, limit=self.settings["lookback"])
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades',
                'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df.set_index('datetime')
        except Exception as e:
            print(f"API Error for {symbol} {interval}: {str(e)}")
            return None

    def calculate_technical_indicators(self, df):
        if df is None or len(df) < 50:
            return None
        df['MA20'] = ta.sma(df['close'], length=20)
        df['MA50'] = ta.sma(df['close'], length=50)
        df['MA100'] = ta.sma(df['close'], length=100)
        df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        bb = ta.bbands(df['close'], length=20)
        df['BB_upper'] = bb['BBU_20_2.0']
        df['BB_lower'] = bb['BBL_20_2.0']
        df['RSI'] = ta.rsi(df['close'], length=14)
        df['MACD'] = ta.macd(df['close'])['MACD_12_26_9']
        df['VWAP'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        df['Volume_MA20'] = df['volume'].rolling(20).mean()
        df['Resistance'] = df['high'].rolling(10).max()
        df['Support'] = df['low'].rolling(10).min()
        return df.dropna()

    def calculate_profit(self, entry, tp):
        gain_pct = (tp - entry) / entry
        profit = gain_pct * self.settings["capital"] * self.settings["leverage"]
        return round(profit, 2), profit >= self.settings["target_profit"]

    def estimate_exit_time(self, df, entry_price, tp1, timeframe):
        df = df.copy()
        df['candle_gain'] = (df['close'] - df['open']) / df['open']
        avg_gain = df[df['candle_gain'] > 0]['candle_gain'].mean()
        if not avg_gain or avg_gain <= 0:
            return None
        required_gain = (tp1 - entry_price) / entry_price
        estimated_candles = required_gain / avg_gain
        if timeframe == "15m":
            return df.index[-1] + timedelta(minutes=15 * estimated_candles)
        elif timeframe == "1h":
            return df.index[-1] + timedelta(hours=estimated_candles)
        else:
            return df.index[-1] + timedelta(hours=4 * estimated_candles)

    def detect_breakout_signal(self, df, symbol, timeframe):
        if df is None or len(df) < 50:
            return None
        last = df.iloc[-1]
        prev = df.iloc[-2]
        price_change = (last['close'] - prev['close']) / prev['close']
        volume_ratio = last['volume'] / df['Volume_MA20'].iloc[-1]
        candle_body = (last['close'] - last['open']) / last['open']
        atr_ratio = last['ATR'] / last['close']
        conditions_met = all([
            price_change >= self.settings["price_change_threshold"],
            volume_ratio >= self.settings["min_volume_ratio"],
            candle_body >= self.settings["min_candle_body"],
            last['close'] > last['Resistance'],
            self.settings["rsi_range"][0] < last['RSI'] < self.settings["rsi_range"][1],
            last['close'] > last['MA20'] > last['MA50'],
            atr_ratio > 0.015
        ])
        if not conditions_met:
            return None
        entry = last['close']
        sl = max(last['Support'] * 0.99, entry - (last['ATR'] * self.settings["atr_multiplier"]))
        risk = entry - sl
        tp1 = entry + risk * 1.618
        profit, valid = self.calculate_profit(entry, tp1)
        if not valid:
            return None
        exit_time = self.estimate_exit_time(df, entry, tp1, timeframe)
        if not exit_time:
            exit_time = last.name + timedelta(hours=2)
        direction = "BUY" if candle_body > 0 else "SELL"
        confidence = round(min(100, 25 * (price_change / self.settings["price_change_threshold"]) +
                                  20 * (volume_ratio / self.settings["min_volume_ratio"]) +
                                  20 * (candle_body / self.settings["min_candle_body"]) +
                                  15 * (last['RSI'] / 70) +
                                  10 * (atr_ratio / 0.02) +
                                  10 * ((last['close'] - last['Resistance']) / last['Resistance'] * 100)), 2)
        signal = {
            'symbol': symbol,
            'timeframe': timeframe,
            'entry': round(entry, 5),
            'sl': round(sl, 5),
            'tp1': round(tp1, 5),
            'confidence': confidence,
            'profit': profit,
            'exit_time': exit_time.strftime('%Y-%m-%d %H:%M:%S'),
            'direction': direction
        }
        self.log_to_csv(signal)
        return signal

    def log_to_csv(self, signal):
        with open(self.csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                signal['symbol'], signal['timeframe'], signal['direction'], signal['entry'],
                signal['tp1'], signal['sl'], signal['confidence'], signal['profit'], signal['exit_time']
            ])

    def send_alert(self, signal):
        message = f"""
🔔 {signal['symbol']} Breakout Signal
Timeframe: {signal['timeframe']} | Direction: {signal['direction']}
Entry: {signal['entry']} | SL: {signal['sl']} | TP1: {signal['tp1']}
Confidence: {signal['confidence']} | Profit: ${signal['profit']}
Exit Estimate: {signal['exit_time']}
"""
        print(message)
        self.bot.send_message(chat_id=self.chat_id, text=message)

    def scan_markets(self):
        print("Starting Breakout Scanner...")
        while True:
            print(f"\nScanning at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            for symbol in self.symbols:
                for tf_name, tf_value in self.timeframes.items():
                    try:
                        df = self.fetch_market_data(symbol, tf_value)
                        df = self.calculate_technical_indicators(df)
                        signal = self.detect_breakout_signal(df, symbol, tf_name)
                        if signal:
                            self.send_alert(signal)
                    except Exception as e:
                        print(f"Error processing {symbol} {tf_name}: {str(e)}")
            time.sleep(60)

if __name__ == "__main__":
    scanner = BreakoutScanner()
    scanner.scan_markets()
