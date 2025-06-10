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
        self.bot = Bot(token="7532851212:AAHOVx1esNWrtk2SJbBQHCXMae7Y-dKJR5o")
        self.client = Client("jU1j7eGK0QJayDX3WwaVcFHOuJfUO3MFU5J9ppfT6DiUslaAkwJrC6mf9lCXBVyU", "aTQtuh5bFKsGxC3bJOPbL7IV4KC1DsQowxgbrO6awi4YbnfCp3BRU55SA58jhRXy")
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
            "1h": Client.KLINE_INTERVAL_1HOUR
        }

        self.settings = {
            "lookback": 1000,
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

    def detect_big_entry_now(self, df, symbol, timeframe):
        if df is None or len(df) < 100:
            return None

        last = df.iloc[-1]
        prev_5_avg = df['close'].iloc[-6:-1].mean()
        price_spike = (last['close'] - prev_5_avg) / prev_5_avg
        volume_spike = last['volume'] / df['Volume_MA20'].iloc[-1]

        if price_spike > 0.05 and volume_spike > 2:
            entry_price = last['close']
            sl = df['low'].iloc[-10:-1].min()
            tp = entry_price + (entry_price - sl) * 2
            exit_time = self.estimate_exit_time(df, entry_price, tp, timeframe)

            signal = {
                'symbol': symbol,
                'timeframe': timeframe,
                'entry': round(entry_price, 5),
                'sl': round(sl, 5),
                'tp1': round(tp, 5),
                'profit': round((tp - entry_price) * self.settings["leverage"], 2),
                'confidence': round(price_spike * 100, 2),
                'exit_time': exit_time.strftime('%Y-%m-%d %H:%M:%S') if exit_time else "N/A",
                'direction': "BUY"
            }

            self.send_alert(signal)
            self.log_to_csv(signal)
            print(f"🔥 BIG ENTRY ALERT for {symbol} on {timeframe}")
            return signal
        return None

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
        print("🚀 Starting Breakout + Big Entry Scanner...")
        while True:
            print(f"\n🕒 Scanning at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            for symbol in self.symbols:
                for tf_name, tf_value in self.timeframes.items():
                    try:
                        df = self.fetch_market_data(symbol, tf_value)
                        df = self.calculate_technical_indicators(df)
                        self.detect_big_entry_now(df, symbol, tf_name)
                    except Exception as e:
                        print(f"Error processing {symbol} {tf_name}: {str(e)}")
            time.sleep(60)

if __name__ == "__main__":
    scanner = BreakoutScanner()
    scanner.scan_markets()
