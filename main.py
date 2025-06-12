import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from binance.client import Client
from ta.trend import EMAIndicator, ADXIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import VolumeWeightedAveragePrice

# === CONFIG ===
API_KEY = ''
API_SECRET = ''
BOT_TOKEN = '7532851212:AAHOVx1esNWrtk2SJbBQHCXMae7Y-dKJR5o'
CHAT_ID = '6494844619'
INTERVAL = Client.KLINE_INTERVAL_1HOUR
LIMIT = 1000  # Increased for better trend analysis

symbols_to_check = [
    "APTUSDT",
    "RPLUSDT",
    "MKRUSDT",
    "ETCUSDT",
    "POPCATUSDT",
    "LAUSDT",
    "PENGUUSDT",
    "HUMAUSDT",
    "AUSDT",
    "AI16ZUSDT",
    "1000BONKUSDT",
    "XLMUSDT",
    "KAITOUSDT",
    "NXPCUSDT",
    "SQDUSDT",
    "WCTUSDT",
    "JUPUSDT",
    "INITUSDT",
    "AIXBTUSDT",
    "GALAUSDT",
    "TONUSDT",
    "HBARUSDT",
    "ORDIUSDT",
    "ICXUSDT",
    "CAKEUSDT",
    "SOPHUSDT",
    "OMUSDT",
    "COOKIEUSDT",
    "1000FLOKIUSDT",
    "GOATUSDT",
    "POLUSDT",
    "PENDLEUSDT",
    "ICPUSDT",
    "PEOPLEUSDT",
    "BMTUSDT",
    "SOONUSDT",
    "SUSDT",
    "ATOMUSDT",
    "DYDXUSDT",
    "RENDERUSDT",
    "USUALUSDT",
    "MOVEUSDT",
    "SKATEUSDT",
    "AXSUSDT",
    "SSVUSDT",
    "MUBARAKUSDT",
    "TAIKOUSDT",
    "GRASSUSDT",
    "BOMEUSDT",
    "UMAUSDT",
    "SEIUSDT",
    "ALGOUSDT",
    "ZROUSDT",
    "CHILLGUYUSDT",
    "TURBOUSDT",
    "1000000MOGUSDT",
    "FLMUSDT",
    "SANDUSDT",
    "RAYSOLUSDT",
    "APEUSDT",
    "CYBERUSDT",
    "SUSHIUSDT",
    "GRIFFAINUSDT",
    "ZENUSDT",
    "IMXUSDT",
    "RUNEUSDT",
    "ARKMUSDT",
    "BIDUSDT",
    "AERGOUSDT",
    "1000SATSUSDT",
    "BUSDT",
    "LAYERUSDT",
    "STXUSDT",
    "BERAUSDT",
    "BRETTUSDT",
    "MERLUSDT",
    "ARCUSDT",
    "OBOLUSDT",
    "LQTYUSDT",
    "VETUSDT",
    "B2USDT",
    "XMRUSDT",
    "HAEDALUSDT",
    "PYTHUSDT",
    "STRKUSDT",
    "IOSTUSDT",
    "ARUSDT",
    "EPTUSDT",
    "VANAUSDT",
    "NOTUSDT",
    "CHESSUSDT",
    "SXTUSDT",
    "BANANAUSDT",
    "KERNELUSDT",
    "DOGSUSDT",
    "GUNUSDT",
    "ZEREBROUSDT",
    "MELANIAUSDT",
    "KASUSDT",
    "XAIUSDT",
    "NILUSDT",
    "TSTUSDT",
    "NEIROETHUSDT",
    "CETUSUSDT",
    "SWARMSUSDT",
    "GRTUSDT",
    "XTZUSDT",
    "ZKUSDT",
    "THETAUSDT",
    "MEMEUSDT",
    "IOUSDT",
    "VINEUSDT",
    "MEUSDT",
    "BDXNUSDT",
    "ACXUSDT",
    "AVAAIUSDT",
    "NEOUSDT",
    "WUSDT",
    "VVVUSDT",
    "DEXEUSDT",
    "GMTUSDT",
    "ACTUSDT",
    "MEWUSDT",
    "SAGAUSDT",
    "IOTAUSDT",
    "QNTUSDT",
    "ROSEUSDT",
    "MANTAUSDT",
    "SNXUSDT",
    "IOTXUSDT",
    "JASMYUSDT",
    "MANAUSDT",
    "BIOUSDT",
    "FIDAUSDT",
    "MINAUSDT",
    "FXSUSDT",
    "REZUSDT",
    "PORTALUSDT",
    "ALICEUSDT",
    "LISTAUSDT",
    "GASUSDT",
    "KAVAUSDT",
    "CFXUSDT",
    "CELOUSDT",
    "COWUSDT",
    "ATAUSDT",
    "CHZUSDT",
    "ATHUSDT",
    "BABYUSDT",
    "DRIFTUSDT",
    "RSRUSDT",
    "DEEPUSDT",
    "PARTIUSDT",
    "FUNUSDT",
    "SOLVUSDT",
    "USDCUSDT",
    "SHELLUSDT",
    "FLOWUSDT",
    "IPUSDT",
    "ALCHUSDT",
    "ACHUSDT",
    "PLUMEUSDT",
    "SIGNUSDT",
    "1MBABYDOGEUSDT",
    "AIUSDT",
    "TUTUSDT",
    "EGLDUSDT",
    "ZRXUSDT",
    "PIPPINUSDT",
    "ALTUSDT",
    "SIRENUSDT",
    "PIXELUSDT",
    "ZILUSDT",
    "RAREUSDT",
    "FORMUSDT",
    "AEROUSDT",
    "AUCTIONUSDT",
    "SKYAIUSDT",
    "DEGOUSDT",
    "GPSUSDT",
    "BLURUSDT",
    "AGTUSDT",
    "GHSTUSDT",
    "HIPPOUSDT",
    "STOUSDT",
    "1INCHUSDT",
    "ZECUSDT",
    "ORCAUSDT",
    "QTUMUSDT",
    "SUPERUSDT",
    "PONKEUSDT",
    "C98USDT",
    "JELLYJELLYUSDT",
    "WOOUSDT",
    "PROMPTUSDT",
    "MAGICUSDT",
    "SONICUSDT",
    "YGGUSDT",
    "AEVOUSDT",
    "MOVRUSDT",
    "MOCAUSDT",
    "ENJUSDT",
    "WALUSDT",
    "CVCUSDT",
    "CATIUSDT",
    "ALPHAUSDT",
    "ONEUSDT",
    "SXPUSDT",
    "BIGTIMEUSDT",
    "MEMEFIUSDT",
    "EPICUSDT",
    "ETHWUSDT",
    "CGPTUSDT",
    "1000LUNCUSDT",
    "API3USDT",
    "AGLDUSDT",
    "HYPERUSDT",
    "GMXUSDT",
    "BROCCOLI714USDT",
    "OMNIUSDT",
    "1000CATUSDT",
    "COTIUSDT",
    "VANRYUSDT",
    "XVGUSDT",
    "1000RATSUSDT",
    "METISUSDT",
    "BSVUSDT",
    "KSMUSDT",
    "BELUSDT",
    "XCNUSDT",
    "BAKEUSDT",
    "SCRUSDT",
    "EDUUSDT",
    "LUNA2USDT",
    "DASHUSDT",
    "LRCUSDT",
    "HIFIUSDT",
    "MORPHOUSDT",
    "DEGENUSDT",
    "VOXELUSDT",
    "UXLINKUSDT",
    "RLCUSDT",
    "BROCCOLIF3BUSDT",
    "BANANAS31USDT",
    "THEUSDT",
    "PORT3USDT",
    "HOTUSDT",
    "B3USDT",
    "ONTUSDT",
    "ASRUSDT",
    "MBOXUSDT",
    "BBUSDT",
    "BEAMXUSDT",
    "NFPUSDT",
    "DYMUSDT",
    "1000XUSDT",
    "REDUSDT",
    "HIGHUSDT",
    "PHAUSDT",
    "POLYXUSDT",
    "PUNDIXUSDT",
    "MYROUSDT",
    "KDAUSDT",
    "LUMIAUSDT",
    "NMRUSDT",
    "CELRUSDT",
    "ZETAUSDT",
    "SPELLUSDT",
    "TOKENUSDT",
    "BICOUSDT",
    "CTSIUSDT",
    "CKBUSDT",
    "SLERFUSDT",
    "IDUSDT",
    "SUNUSDT",
    "PUFFERUSDT",
    "TUSDT",
    "GTCUSDT",
    "KNCUSDT",
    "ANKRUSDT",
    "DOODUSDT",
    "FISUSDT",
    "HFTUSDT",
    "BRUSDT",
    "STGUSDT",
    "BATUSDT",
    "TLMUSDT",
    "DFUSDT",
    "STORJUSDT",
    "ARKUSDT",
    "JSTUSDT",
    "CHRUSDT",
    "ASTRUSDT",
    "TNSRUSDT",
    "AKTUSDT",
    "FHEUSDT",
    "OGNUSDT",
    "TRUUSDT",
    "QUICKUSDT",
    "DOLOUSDT",
    "VTHOUSDT",
    "SWELLUSDT",
    "RDNTUSDT",
    "1000CHEEMSUSDT",
    "JOEUSDT",
    "SAFEUSDT",
    "PHBUSDT",
    "MAVUSDT",
    "DUSKUSDT",
    "LEVERUSDT",
    "AWEUSDT",
    "SKLUSDT",
    "NKNUSDT",
    "DENTUSDT",
    "ACEUSDT",
    "RONINUSDT",
    "PUMPUSDT",
    "HIVEUSDT",
    "1000XECUSDT",
    "AVAUSDT",
    "BANDUSDT",
    "HOOKUSDT",
    "GLMUSDT",
    "DUSDT",
    "KOMAUSDT",
    "BSWUSDT",
    "PERPUSDT",
    "MLNUSDT",
    "SYNUSDT",
    "AIOTUSDT",
    "VICUSDT",
    "USTCUSDT",
    "TWTUSDT",
    "LOKAUSDT",
    "WAXPUSDT",
    "MTLUSDT",
    "OGUSDT",
    "ONGUSDT",
    "DODOXUSDT",
    "OXTUSDT",
    "ARPAUSDT",
    "PROMUSDT",
    "VELODROMEUSDT",
    "BNTUSDT",
    "ILVUSDT",
    "RIFUSDT",
    "MILKUSDT",
    "XVSUSDT",
    "ZKJUSDT",
    "HEIUSDT",
    "1000WHYUSDT",
    "BANKUSDT",
    "SFPUSDT",
    "REIUSDT",
    "LSKUSDT",
    "GUSDT",
    "POWRUSDT",
    "NTRNUSDT",
    "DIAUSDT",
    "STEEMUSDT",
    "MAVIAUSDT",
    "SYSUSDT",
    "FLUXUSDT",
    "BANUSDT",
    "CTKUSDT",
    "SCRTUSDT",
    "ALPINEUSDT",
    "FIOUSDT"
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
        '0.236': high - 0.236 * diff,
        '0.382': high - 0.382 * diff,
        '0.5': high - 0.5 * diff,
        '0.618': high - 0.618 * diff,
        '0.786': high - 0.786 * diff
    }

def calculate_support_resistance(df):
    # Calculate recent support and resistance using fractal analysis
    sr_levels = []
    for i in range(2, len(df)-2):
        if df['low'].iloc[i] < df['low'].iloc[i-1] and df['low'].iloc[i] < df['low'].iloc[i-2] and \
           df['low'].iloc[i] < df['low'].iloc[i+1] and df['low'].iloc[i] < df['low'].iloc[i+2]:
            sr_levels.append(df['low'].iloc[i])
        elif df['high'].iloc[i] > df['high'].iloc[i-1] and df['high'].iloc[i] > df['high'].iloc[i-2] and \
             df['high'].iloc[i] > df['high'].iloc[i+1] and df['high'].iloc[i] > df['high'].iloc[i+2]:
            sr_levels.append(df['high'].iloc[i])
    
    # Cluster nearby levels
    clusters = []
    for level in sorted(sr_levels):
        if not clusters:
            clusters.append([level])
        else:
            last_cluster = clusters[-1]
            if abs(level - np.mean(last_cluster)) < 0.005 * np.mean(last_cluster):  # 0.5% threshold
                last_cluster.append(level)
            else:
                clusters.append([level])
    
    # Return strongest support/resistance levels (most touches)
    sr_points = [np.mean(cluster) for cluster in clusters]
    sr_strength = [len(cluster) for cluster in clusters]
    
    # Sort by strength and return top 3
    sorted_indices = np.argsort(sr_strength)[::-1]
    return [sr_points[i] for i in sorted_indices[:3]]

def predict_signal(df):
    # Calculate indicators
    df['EMA20'] = EMAIndicator(df['close'], window=20).ema_indicator()
    df['EMA50'] = EMAIndicator(df['close'], window=50).ema_indicator()
    df['EMA100'] = EMAIndicator(df['close'], window=100).ema_indicator()
    df['RSI'] = RSIIndicator(df['close'], window=14).rsi()
    df['Stoch_%K'] = StochasticOscillator(df['high'], df['low'], df['close'], window=14).stoch()
    df['Stoch_%D'] = df['Stoch_%K'].rolling(3).mean()
    df['ATR'] = AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    df['ADX'] = ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()
    df['VWAP'] = VolumeWeightedAveragePrice(df['high'], df['low'], df['close'], df['volume'], window=20).volume_weighted_average_price()
    
    # MACD
    macd = MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
    
    # Bollinger Bands
    bb = BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_width'] = df['bb_upper'] - df['bb_lower']
    
    # Recent price action
    recent_high = df['high'].iloc[-100:-5].max()
    recent_low = df['low'].iloc[-100:-5].min()
    latest_close = df['close'].iloc[-1]
    latest_low = df['low'].iloc[-1]
    latest_high = df['high'].iloc[-1]
    
    # Volume analysis
    latest_volume = df['volume'].iloc[-1]
    volume_avg = df['volume'].rolling(50).mean().iloc[-1]
    volume_std = df['volume'].rolling(50).std().iloc[-1]
    
    # Indicator values
    rsi = df['RSI'].iloc[-1]
    stoch_k = df['Stoch_%K'].iloc[-1]
    stoch_d = df['Stoch_%D'].iloc[-1]
    atr = df['ATR'].iloc[-1]
    adx = df['ADX'].iloc[-1]
    macd_hist = df['MACD_Hist'].iloc[-1]
    vwap = df['VWAP'].iloc[-1]
    
    # Support/resistance analysis
    sr_levels = calculate_support_resistance(df)
    near_support = any(abs(latest_close - level) < 0.008 * level for level in sr_levels)
    near_resistance = any(abs(latest_close - level) < 0.008 * level for level in sr_levels)
    below_vwap = latest_close < vwap
    above_vwap = latest_close > vwap
    
    # ===== BEARISH CONDITIONS (SELL/SHORT) =====
    squeeze = df['bb_width'].iloc[-1] < df['bb_width'].rolling(100).mean().iloc[-1] * 0.6
    momentum_down = df['close'].iloc[-1] < df['close'].iloc[-5]
    ma_bearish = df['EMA20'].iloc[-1] < df['EMA50'].iloc[-1] < df['EMA100'].iloc[-1]
    ma_cross_death = df['EMA20'].iloc[-2] > df['EMA50'].iloc[-2] and df['EMA20'].iloc[-1] < df['EMA50'].iloc[-1]
    
    # Volume analysis
    volume_spike = latest_volume > volume_avg + 1.5 * volume_std
    volume_divergence = (latest_close < df['close'].iloc[-2]) and (latest_volume > df['volume'].iloc[-2])
    
    # RSI/Stochastic conditions
    rsi_down = rsi < 45 and df['RSI'].iloc[-1] < df['RSI'].iloc[-2]
    stoch_down = stoch_k < 40 and stoch_k < stoch_d and df['Stoch_%K'].iloc[-1] < df['Stoch_%K'].iloc[-2]
    
    # MACD conditions
    macd_down = macd_hist < 0 and df['MACD_Hist'].iloc[-1] < df['MACD_Hist'].iloc[-2]
    
    # ===== BULLISH CONDITIONS (BUY/LONG) =====
    momentum_up = df['close'].iloc[-1] > df['close'].iloc[-5]
    ma_bullish = df['EMA20'].iloc[-1] > df['EMA50'].iloc[-1] > df['EMA100'].iloc[-1]
    ma_cross_golden = df['EMA20'].iloc[-2] < df['EMA50'].iloc[-2] and df['EMA20'].iloc[-1] > df['EMA50'].iloc[-1]
    
    # RSI/Stochastic conditions
    rsi_up = rsi > 55 and df['RSI'].iloc[-1] > df['RSI'].iloc[-2]
    stoch_up = stoch_k > 60 and stoch_k > stoch_d and df['Stoch_%K'].iloc[-1] > df['Stoch_%K'].iloc[-2]
    
    # MACD conditions
    macd_up = macd_hist > 0 and df['MACD_Hist'].iloc[-1] > df['MACD_Hist'].iloc[-2]
    
    # Support/resistance breakout
    resistance_break = latest_close > max(sr_levels) if sr_levels else False
    support_break = latest_close < min(sr_levels) if sr_levels else False
    
    # ===== CONFIDENCE SCORING =====
    bearish_confidence = 0
    if near_support: bearish_confidence += 25
    if volume_spike: bearish_confidence += 20
    if momentum_down: bearish_confidence += 15
    if ma_bearish: bearish_confidence += 10
    if ma_cross_death: bearish_confidence += 15
    if rsi_down: bearish_confidence += 10
    if stoch_down: bearish_confidence += 10
    if macd_down: bearish_confidence += 10
    if squeeze: bearish_confidence += 10
    if adx > 25: bearish_confidence += 15
    if below_vwap: bearish_confidence += 10
    
    bullish_confidence = 0
    if resistance_break: bullish_confidence += 30
    if volume_spike: bullish_confidence += 20
    if momentum_up: bullish_confidence += 15
    if ma_bullish: bullish_confidence += 10
    if ma_cross_golden: bullish_confidence += 15
    if rsi_up: bullish_confidence += 10
    if stoch_up: bullish_confidence += 10
    if macd_up: bullish_confidence += 10
    if adx > 25: bullish_confidence += 15
    if above_vwap: bullish_confidence += 10
    
    # Cap at 100
    bearish_confidence = min(bearish_confidence, 100)
    bullish_confidence = min(bullish_confidence, 100)
    
    fib_levels = calculate_fibonacci_levels(recent_high, recent_low)
    
    # ===== DECISION MAKING =====
    if bearish_confidence >= 75 and bearish_confidence > bullish_confidence:
        # SELL/SHORT signal
        entry_price = round(latest_close, 4)
        sl = round(entry_price + 1.5 * atr, 4)
        tp1 = round(entry_price - 1 * atr, 4)
        tp2 = round(entry_price - 2 * atr, 4)
        tp3 = round(entry_price - 3 * atr, 4)
        
        return {
            "signal": "SELL",
            "entry_price": entry_price,
            "entry_time": df.index[-1],
            "predicted_exit_time": df.index[-1] + timedelta(hours=6),
            "sl": sl,
            "tp": [tp1, tp2, tp3],
            "confidence_score": f"{bearish_confidence}%",
            "fibonacci_levels": fib_levels,
            "support_levels": [round(x, 4) for x in sr_levels],
            "indicators": {
                "rsi": round(rsi, 2),
                "stoch": f"{round(stoch_k, 2)}/{round(stoch_d, 2)}",
                "atr": round(atr, 4),
                "adx": round(adx, 2),
                "macd_hist": round(macd_hist, 4),
                "volume_ratio": round(latest_volume / volume_avg, 2),
                "ma_trend": "Bearish" if ma_bearish else "Neutral",
                "ma_cross": "Death Cross" if ma_cross_death else "None",
                "squeeze": squeeze,
                "vwap_position": "Below" if below_vwap else "Above"
            }
        }
    
    elif bullish_confidence >= 75:
        # BUY/LONG signal
        entry_price = round(latest_close, 4)
        sl = round(entry_price - 1.5 * atr, 4)
        tp1 = round(entry_price + 1 * atr, 4)
        tp2 = round(entry_price + 2 * atr, 4)
        tp3 = round(entry_price + 3 * atr, 4)
        
        return {
            "signal": "BUY",
            "entry_price": entry_price,
            "entry_time": df.index[-1],
            "predicted_exit_time": df.index[-1] + timedelta(hours=6),
            "sl": sl,
            "tp": [tp1, tp2, tp3],
            "confidence_score": f"{bullish_confidence}%",
            "fibonacci_levels": fib_levels,
            "support_levels": [round(x, 4) for x in sr_levels],
            "indicators": {
                "rsi": round(rsi, 2),
                "stoch": f"{round(stoch_k, 2)}/{round(stoch_d, 2)}",
                "atr": round(atr, 4),
                "adx": round(adx, 2),
                "macd_hist": round(macd_hist, 4),
                "volume_ratio": round(latest_volume / volume_avg, 2),
                "ma_trend": "Bullish" if ma_bullish else "Neutral",
                "ma_cross": "Golden Cross" if ma_cross_golden else "None",
                "squeeze": squeeze,
                "vwap_position": "Above" if above_vwap else "Below"
            }
        }
    
    else:
        # No clear signal
        breakdown_prob = bearish_confidence / 100
        breakout_prob = bullish_confidence / 100
        
        # Estimate time to potential signal
        estimated_hours = 12  # Default
        if max(bearish_confidence, bullish_confidence) > 50:
            estimated_hours = 6
        
        predicted_time = datetime.now() + timedelta(hours=estimated_hours)
        predicted_price_down = round(recent_low * (1 - 0.01 * breakdown_prob), 4)
        predicted_price_up = round(recent_high * (1 + 0.01 * breakout_prob), 4)
        
        return {
            "signal": "HOLD",
            "confidence_score": f"Bullish: {bullish_confidence}% | Bearish: {bearish_confidence}%",
            "forecast": {
                "expected_time": predicted_time,
                "expected_price_down": predicted_price_down,
                "expected_price_up": predicted_price_up
            },
            "fibonacci_levels": fib_levels,
            "support_levels": [round(x, 4) for x in sr_levels],
            "next_check": datetime.now() + timedelta(minutes=30)
        }

def format_message(symbol, result):
    if result['signal'] == "SELL":
        msg = f"""
🔴 *SELL SIGNAL* — `{symbol}` 🔴
📉 *Type*: Short Position (Bearish Breakdown)
🔻 *Confidence*: {result['confidence_score']}

🎯 Entry Price: {result['entry_price']}
⏰ Entry Time: {result['entry_time'].strftime('%Y-%m-%d %H:%M')}
📆 Expected Duration: ~6 hours

🛑 Stop Loss: {result['sl']}
💰 Take Profit Targets:
   - TP1: {result['tp'][0]} (1:1)
   - TP2: {result['tp'][1]} (2:1)
   - TP3: {result['tp'][2]} (3:1)

📊 Indicators:
   - RSI: {result['indicators']['rsi']} (Bearish)
   - Stoch: {result['indicators']['stoch']} (Bearish)
   - MACD Hist: {result['indicators']['macd_hist']} (Negative)
   - ADX: {result['indicators']['adx']} (Strong Trend)
   - Volume: x{result['indicators']['volume_ratio']} (Spike)
   - MA Trend: {result['indicators']['ma_trend']} {result['indicators']['ma_cross']}
   - VWAP: {result['indicators']['vwap_position']}
   - BB Squeeze: {result['indicators']['squeeze']}

📉 Support Levels: {', '.join(map(str, result['support_levels']))}
📐 Fibonacci Levels:
   0.236 = {result['fibonacci_levels']['0.236']:.4f}
   0.382 = {result['fibonacci_levels']['0.382']:.4f}
   0.5 = {result['fibonacci_levels']['0.5']:.4f}
   0.618 = {result['fibonacci_levels']['0.618']:.4f}
   0.786 = {result['fibonacci_levels']['0.786']:.4f}

⚠️ Risk Warning: Always use proper risk management. Suggested position size 1-2% of capital.
"""
    elif result['signal'] == "BUY":
        msg = f"""
🟢 *BUY SIGNAL* — `{symbol}` 🟢
📈 *Type*: Long Position (Bullish Breakout)
🔼 *Confidence*: {result['confidence_score']}

🎯 Entry Price: {result['entry_price']}
⏰ Entry Time: {result['entry_time'].strftime('%Y-%m-%d %H:%M')}
📆 Expected Duration: ~6 hours

🛑 Stop Loss: {result['sl']}
💰 Take Profit Targets:
   - TP1: {result['tp'][0]} (1:1)
   - TP2: {result['tp'][1]} (2:1)
   - TP3: {result['tp'][2]} (3:1)

📊 Indicators:
   - RSI: {result['indicators']['rsi']} (Bullish)
   - Stoch: {result['indicators']['stoch']} (Bullish)
   - MACD Hist: {result['indicators']['macd_hist']} (Positive)
   - ADX: {result['indicators']['adx']} (Strong Trend)
   - Volume: x{result['indicators']['volume_ratio']} (Spike)
   - MA Trend: {result['indicators']['ma_trend']} {result['indicators']['ma_cross']}
   - VWAP: {result['indicators']['vwap_position']}
   - BB Squeeze: {result['indicators']['squeeze']}

📈 Resistance Levels: {', '.join(map(str, result['support_levels']))}
📐 Fibonacci Levels:
   0.236 = {result['fibonacci_levels']['0.236']:.4f}
   0.382 = {result['fibonacci_levels']['0.382']:.4f}
   0.5 = {result['fibonacci_levels']['0.5']:.4f}
   0.618 = {result['fibonacci_levels']['0.618']:.4f}
   0.786 = {result['fibonacci_levels']['0.786']:.4f}

⚠️ Risk Warning: Always use proper risk management. Suggested position size 1-2% of capital.
"""
    else:
        msg = f"""
⚪ *HOLD* — `{symbol}`
📊 *Market Status*: No clear signal
🔸 Bullish Confidence: {result['confidence_score'].split('|')[0]}
🔹 Bearish Confidence: {result['confidence_score'].split('|')[1]}

⏳ Next Potential Movement:
   - Expected Time: {result['forecast']['expected_time'].strftime('%Y-%m-%d %H:%M')}
   - Upside Target: {result['forecast']['expected_price_up']}
   - Downside Target: {result['forecast']['expected_price_down']}

📊 Key Levels:
   - Support: {min(result['support_levels']):.4f}
   - Resistance: {max(result['support_levels']):.4f}

🔄 Next Analysis: {result['next_check'].strftime('%H:%M')}
"""
    return msg.strip()

# Main loop
while True:
    for symbol in symbols_to_check:
        try:
            df = fetch_binance_data(symbol)
            if df is None or len(df) < 200:
                continue

            result = predict_signal(df)
            msg = format_message(symbol, result)
            
            # Only send alerts for high confidence signals or if the status changed significantly
            if result['signal'] in ["BUY", "SELL"]:
                if symbol not in last_alerts or last_alerts[symbol]['signal'] != result['signal']:
                    send_telegram_message(msg)
                    print(f"[ALERT] {symbol} {result['signal']} signal detected")
                    last_alerts[symbol] = result
            else:
                # Only send monitoring updates every 6 hours for the same status
                if symbol not in last_alerts or datetime.now() - last_alerts[symbol]['time'] > timedelta(hours=6):
                    send_telegram_message(msg)
                    print(f"[{symbol}] Monitoring - no clear signal")
                    last_alerts[symbol] = {'signal': "HOLD", 'time': datetime.now()}
            
            time.sleep(1.5)  # Rate limit
            
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
            time.sleep(5)
    
    # Wait before next full cycle
    print(f"Completed full scan at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    time.sleep(60 * 30)  # Wait 15 minutes before next scan