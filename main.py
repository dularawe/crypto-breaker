import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from binance.client import Client
from ta.trend import EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange, BollingerBands

# === CONFIG ===
API_KEY = ''
API_SECRET = ''
BOT_TOKEN = '7532851212:AAHOVx1esNWrtk2SJbBQHCXMae7Y-dKJR5o'
CHAT_ID = '7771111812'
INTERVAL = Client.KLINE_INTERVAL_1HOUR
LIMIT = 900

symbols_to_check = [
    "NMRUSDT", "UMAUSDT", "PHAUSDT", "ZENUSDT", "ENSUSDT",
    "ILVUSDT", "FXSUSDT", "CVCUSDT", "KNCUSDT", "ASTRUSDT",
    "LPTUSDT", "REQUSDT", "ANKRUSDT", "ALPHAUSDT", "BANDUSDT",
    "RIFUSDT", "MDTUSDT", "KEYUSDT", "CTSIUSDT", "CELRUSDT",
    "GLMRUSDT", "ARUSDT"
]

client = Client(API_KEY, API_SECRET)
last_alerts = {}

def send_telegram_message(text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": text, "parse_mode": "Markdown"}
    try:
        requests.post(url, data=data)
    except Exception as e:
        print("Telegram Error:", e)

def fetch_binance_data(symbol):
    try:
        klines = client.get_klines(symbol=symbol, interval=INTERVAL, limit=LIMIT)
        df = pd.DataFrame(klines, columns=[
            'time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'qav', 'trades', 'tbbav', 'tbqav', 'ignore'
        ])
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        df.set_index('time', inplace=True)
        df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        return df
    except Exception as e:
        print(f"Fetch error for {symbol}: {e}")
        return None

def calculate_fibonacci_levels(high, low):
    diff = high - low
    return {
        '0.236': round(high - 0.236 * diff, 4),
        '0.382': round(high - 0.382 * diff, 4),
        '0.5': round(high - 0.5 * diff, 4),
        '0.618': round(high - 0.618 * diff, 4),
        '0.786': round(high - 0.786 * diff, 4)
    }

def predict_breakdown(df):
    df['EMA20'] = EMAIndicator(df['close'], window=20).ema_indicator()
    df['EMA50'] = EMAIndicator(df['close'], window=50).ema_indicator()
    df['RSI'] = RSIIndicator(df['close'], window=14).rsi()
    df['ATR'] = AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    df['ADX'] = ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()
    bb = BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_width'] = bb.bollinger_hband() - bb.bollinger_lband()

    recent_high = df['high'].iloc[-100:-5].max()
    recent_low = df['low'].iloc[-100:-5].min()
    latest_close = df['close'].iloc[-1]
    latest_volume = df['volume'].iloc[-1]
    volume_avg = df['volume'].iloc[-100:-1].mean()
    rsi = df['RSI'].iloc[-1]
    atr = df['ATR'].iloc[-1]
    adx = df['ADX'].iloc[-1]
    squeeze = df['bb_width'].iloc[-1] < df['bb_width'].rolling(100).mean().iloc[-1] * 0.6
    momentum = df['close'].iloc[-1] < df['close'].iloc[-10]
    ma_bearish = df['EMA20'].iloc[-1] < df['EMA50'].iloc[-1]
    near_support = (latest_close - recent_low) / latest_close < 0.008
    volume_spike = latest_volume > volume_avg * 1.3
    rsi_down = rsi < 45

    fib_levels = calculate_fibonacci_levels(recent_high, recent_low)

    confidence = 0
    if near_support: confidence += 20
    if volume_spike: confidence += 20
    if momentum: confidence += 15
    if ma_bearish: confidence += 10
    if rsi_down: confidence += 10
    if squeeze: confidence += 10
    if adx > 25: confidence += 15

    if near_support and volume_spike and momentum:
        entry_price = round(latest_close, 4)
        sl = round(entry_price + 1.5 * atr, 4)
        tp = round(entry_price - 2.5 * atr, 4)
        return {
            "status": "BREAKDOWN STARTED",
            "entry_price": entry_price,
            "entry_time": df.index[-1],
            "predicted_exit_time": df.index[-1] + timedelta(hours=3),
            "sl": sl,
            "tp": tp,
            "confidence_score": f"{confidence}%",
            "fibonacci_levels": fib_levels,
            "indicators": {
                "rsi": round(rsi, 2),
                "atr": round(atr, 4),
                "adx": round(adx, 2),
                "volume_ratio": round(latest_volume / volume_avg, 2),
                "ma_trend": "Bearish" if ma_bearish else "Neutral",
                "squeeze": squeeze
            }
        }
    else:
        predicted_time = datetime.now() + timedelta(hours=2)
        predicted_price = round(recent_low * 0.995, 4)
        return {
            "status": "NO BREAKDOWN YET",
            "confidence_score": f"{confidence}%",
            "forecast": {
                "expected_time": predicted_time,
                "expected_price": predicted_price
            },
            "fibonacci_levels": fib_levels
        }

for symbol in symbols_to_check:
    df = fetch_binance_data(symbol)
    if df is None or len(df) < 200:
        continue

    result = predict_breakdown(df)

    if result['status'] == "BREAKDOWN STARTED":
        msg = f"""
🚨 *Breakdown Detected* — `{symbol}`
📉 Entry: {result['entry_price']}
📈 SL: {result['sl']} | 📉 TP: {result['tp']}
⏰ Entry Time: {result['entry_time'].strftime('%Y-%m-%d %H:%M')}
📆 Exit Time: {result['predicted_exit_time'].strftime('%Y-%m-%d %H:%M')}
📊 RSI: {result['indicators']['rsi']} | ATR: {result['indicators']['atr']}
📊 ADX: {result['indicators']['adx']} | Vol x{result['indicators']['volume_ratio']}
📉 MA Trend: {result['indicators']['ma_trend']} | Squeeze: {result['indicators']['squeeze']}
🔐 Confidence: {result['confidence_score']}
📐 Fib Levels: 0.236={result['fibonacci_levels']['0.236']}, 0.382={result['fibonacci_levels']['0.382']}, 0.5={result['fibonacci_levels']['0.5']}, 0.618={result['fibonacci_levels']['0.618']}, 0.786={result['fibonacci_levels']['0.786']}
        """
        send_telegram_message(msg.strip())
        print(f"[ALERT] {symbol} breakdown detected")
    else:
        msg = f"""
🕵️ *Monitoring* — `{symbol}`
⏰ Predicted Breakdown: {result['forecast']['expected_time'].strftime('%Y-%m-%d %H:%M')}
🎯 Target Price: {result['forecast']['expected_price']}
🔐 Confidence: {result['confidence_score']}
📐 Fib Levels: 0.236={result['fibonacci_levels']['0.236']}, 0.382={result['fibonacci_levels']['0.382']}, 0.5={result['fibonacci_levels']['0.5']}, 0.618={result['fibonacci_levels']['0.618']}, 0.786={result['fibonacci_levels']['0.786']}
        """
        send_telegram_message(msg.strip())
        print(f"[{symbol}] Watching for breakdown.")
    time.sleep(1.5)
