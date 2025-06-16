import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from binance.client import Client
from ta.trend import EMAIndicator, ADXIndicator, MACD, IchimokuIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, AwesomeOscillatorIndicator
from ta.volatility import AverageTrueRange, BollingerBands, KeltnerChannel
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator
from ta.others import DailyReturnIndicator, CumulativeReturnIndicator

# === CONFIG ===
API_KEY = ''
API_SECRET = ''
BOT_TOKEN = '7532851212:AAHOVx1esNWrtk2SJbBQHCXMae7Y-dKJR5o'
CHAT_IDS = [
    '6494844619',  # First chat ID
    '1265683834',
    '7771111812'   # Second chat ID
]
INTERVAL = Client.KLINE_INTERVAL_1HOUR
LIMIT = 3000  # Optimized for better performance

# Enhanced symbol list with categorization
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
early_warnings = {}

def send_telegram_message(text: str) -> None:
    """Broadcast *text* to every CHAT_ID in CHAT_IDS with Markdown parsing."""
    for chat_id in CHAT_IDS:
        payload = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        }
        try:
            requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage", data=payload, timeout=10)
        except Exception as exc:
            print(f"[Telegram] Failed to send to {chat_id}: {exc}")

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
        '0.786': high - 0.786 * diff,
        '1.0': high,
        '1.272': high + 0.272 * diff,
        '1.618': high + 0.618 * diff
    }

def calculate_support_resistance(df, window=50):
    """Improved S/R detection using clustering and volume profile"""
    levels = []
    
    # Fractal levels
    for i in range(2, len(df)-2):
        # Support fractals
        if df['low'].iloc[i] < df['low'].iloc[i-1] and df['low'].iloc[i] < df['low'].iloc[i-2] and \
           df['low'].iloc[i] < df['low'].iloc[i+1] and df['low'].iloc[i] < df['low'].iloc[i+2]:
            levels.append((df['low'].iloc[i], df['volume'].iloc[i]))
        # Resistance fractals
        elif df['high'].iloc[i] > df['high'].iloc[i-1] and df['high'].iloc[i] > df['high'].iloc[i-2] and \
             df['high'].iloc[i] > df['high'].iloc[i+1] and df['high'].iloc[i] > df['high'].iloc[i+2]:
            levels.append((df['high'].iloc[i], df['volume'].iloc[i]))
    
    # Pivot points
    pivot = (df['high'].iloc[-1] + df['low'].iloc[-1] + df['close'].iloc[-1]) / 3
    levels.append((pivot, np.mean(df['volume'].iloc[-3:])))
    
    # Volume-weighted clustering
    if levels:
        levels_sorted = sorted(levels, key=lambda x: x[0])
        clusters = []
        
        for level in levels_sorted:
            if not clusters:
                clusters.append([level])
            else:
                last_cluster = clusters[-1]
                last_avg = sum(l[0] for l in last_cluster)/len(last_cluster)
                if abs(level[0] - last_avg) < 0.01 * last_avg:  # 1% threshold
                    last_cluster.append(level)
                else:
                    clusters.append([level])
        
        # Calculate cluster strength (volume-weighted)
        sr_points = []
        for cluster in clusters:
            total_volume = sum(v for _, v in cluster)
            weighted_price = sum(p*v for p, v in cluster) / total_volume
            sr_points.append((weighted_price, total_volume))
        
        # Sort by strength and return top 5
        sr_points.sort(key=lambda x: x[1], reverse=True)
        return [x[0] for x in sr_points[:5]]
    
    return []

def detect_early_patterns(df):
    """Detect emerging patterns before they fully form"""
    patterns = {
        'potential_bullish': False,
        'potential_bearish': False,
        'pattern_type': None,
        'expected_confirm_time': None,
        'potential_entry': None
    }
    
    # Check for emerging bullish patterns
    # 1. Potential Inverse Head & Shoulders
    if len(df) > 50:
        left_low = df['low'].iloc[-20:-15].min()
        head_low = df['low'].iloc[-15:-5].min()
        right_low = df['low'].iloc[-5:].min()
        
        if (left_low < head_low and right_low < head_low and 
            abs(left_low - right_low) < 0.01 * head_low and
            df['close'].iloc[-1] > (left_low + head_low + right_low)/3):
            patterns['potential_bullish'] = True
            patterns['pattern_type'] = "Inverse Head & Shoulders"
            patterns['expected_confirm_time'] = df.index[-1] + timedelta(hours=3)
            patterns['potential_entry'] = df['high'].iloc[-15:-5].max()  # Neckline
    
    # 2. Potential Bull Flag
    if len(df) > 30:
        flag_pole = df['high'].iloc[-20] - df['low'].iloc[-20]
        flag_high = df['high'].iloc[-20:-5].max()
        flag_low = df['low'].iloc[-20:-5].min()
        
        if (flag_pole > 0.05 * df['close'].iloc[-20] and  # Significant move
            (flag_high - flag_low) < 0.5 * flag_pole and  # Tight consolidation
            df['volume'].iloc[-20:-5].mean() < df['volume'].iloc[-25:-20].mean() * 0.7 and  # Volume contraction
            df['close'].iloc[-1] > flag_high):  # Breaking out
            patterns['potential_bullish'] = True
            patterns['pattern_type'] = "Bull Flag"
            patterns['expected_confirm_time'] = df.index[-1] + timedelta(hours=2)
            patterns['potential_entry'] = flag_high
    
    # Check for emerging bearish patterns
    # 1. Potential Head & Shoulders
    if len(df) > 50:
        left_high = df['high'].iloc[-20:-15].max()
        head_high = df['high'].iloc[-15:-5].max()
        right_high = df['high'].iloc[-5:].max()
        
        if (left_high > head_high and right_high > head_high and 
            abs(left_high - right_high) < 0.01 * head_high and
            df['close'].iloc[-1] < (left_high + head_high + right_high)/3):
            patterns['potential_bearish'] = True
            patterns['pattern_type'] = "Head & Shoulders"
            patterns['expected_confirm_time'] = df.index[-1] + timedelta(hours=3)
            patterns['potential_entry'] = df['low'].iloc[-15:-5].min()  # Neckline
    
    # 2. Potential Bear Flag
    if len(df) > 30:
        flag_pole = df['high'].iloc[-20] - df['low'].iloc[-20]
        flag_high = df['high'].iloc[-20:-5].max()
        flag_low = df['low'].iloc[-20:-5].min()
        
        if (flag_pole > 0.05 * df['close'].iloc[-20] and  # Significant move
            (flag_high - flag_low) < 0.5 * flag_pole and  # Tight consolidation
            df['volume'].iloc[-20:-5].mean() < df['volume'].iloc[-25:-20].mean() * 0.7 and  # Volume contraction
            df['close'].iloc[-1] < flag_low):  # Breaking down
            patterns['potential_bearish'] = True
            patterns['pattern_type'] = "Bear Flag"
            patterns['expected_confirm_time'] = df.index[-1] + timedelta(hours=2)
            patterns['potential_entry'] = flag_low
    
    return patterns

def predict_early_entry(df):
    """Predict future entry points before signals fully form"""
    early_signals = {
        'potential_entries': [],
        'next_analysis_time': datetime.now() + timedelta(minutes=15)
    }
    
    # Get the emerging patterns
    patterns = detect_early_patterns(df)
    
    # Calculate key levels
    sr_levels = calculate_support_resistance(df)
    fib_levels = calculate_fibonacci_levels(df['high'].max(), df['low'].min())
    
    # Check for potential bullish entries
    if patterns['potential_bullish']:
        entry_info = {
            'type': 'BUY',
            'pattern': patterns['pattern_type'],
            'potential_entry': patterns['potential_entry'],
            'confirm_time': patterns['expected_confirm_time'],
            'stop_loss': patterns['potential_entry'] * 0.99,  # 1% below entry
            'take_profit': patterns['potential_entry'] * 1.03,  # 3% above
            'confidence': "Medium",
            'key_levels': {
                'support': min(sr_levels) if sr_levels else None,
                'resistance': max(sr_levels) if sr_levels else None,
                'fib_618': fib_levels['0.618']
            }
        }
        early_signals['potential_entries'].append(entry_info)
    
    # Check for potential bearish entries
    if patterns['potential_bearish']:
        entry_info = {
            'type': 'SELL',
            'pattern': patterns['pattern_type'],
            'potential_entry': patterns['potential_entry'],
            'confirm_time': patterns['expected_confirm_time'],
            'stop_loss': patterns['potential_entry'] * 1.01,  # 1% above entry
            'take_profit': patterns['potential_entry'] * 0.97,  # 3% below
            'confidence': "Medium",
            'key_levels': {
                'support': min(sr_levels) if sr_levels else None,
                'resistance': max(sr_levels) if sr_levels else None,
                'fib_618': fib_levels['0.618']
            }
        }
        early_signals['potential_entries'].append(entry_info)
    
    # Check for moving average convergences
    ema20 = df['EMA20'].iloc[-1]
    ema50 = df['EMA50'].iloc[-1]
    ema100 = df['EMA100'].iloc[-1]
    
    # Potential Golden Cross
    if (ema20 > ema50 and 
        abs(ema20 - ema50) < 0.005 * ema50 and  # Within 0.5%
        df['EMA20'].iloc[-2] < df['EMA50'].iloc[-2]):  # Was below
        expected_time = df.index[-1] + timedelta(hours=2)
        entry_info = {
            'type': 'BUY',
            'pattern': "Potential Golden Cross",
            'potential_entry': df['close'].iloc[-1],
            'confirm_time': expected_time,
            'stop_loss': df['low'].iloc[-5:].min(),
            'take_profit': df['close'].iloc[-1] * 1.02,
            'confidence': "High",
            'key_levels': {
                'support': min(sr_levels) if sr_levels else None,
                'resistance': max(sr_levels) if sr_levels else None
            }
        }
        early_signals['potential_entries'].append(entry_info)
    
    # Potential Death Cross
    if (ema20 < ema50 and 
        abs(ema20 - ema50) < 0.005 * ema50 and  # Within 0.5%
        df['EMA20'].iloc[-2] > df['EMA50'].iloc[-2]):  # Was above
        expected_time = df.index[-1] + timedelta(hours=2)
        entry_info = {
            'type': 'SELL',
            'pattern': "Potential Death Cross",
            'potential_entry': df['close'].iloc[-1],
            'confirm_time': expected_time,
            'stop_loss': df['high'].iloc[-5:].max(),
            'take_profit': df['close'].iloc[-1] * 0.98,
            'confidence': "High",
            'key_levels': {
                'support': min(sr_levels) if sr_levels else None,
                'resistance': max(sr_levels) if sr_levels else None
            }
        }
        early_signals['potential_entries'].append(entry_info)
    
    return early_signals

def format_early_entry_message(symbol, entry_info):
    msg = f"""
🔍 *POTENTIAL ENTRY ALERT* — `{symbol}`
📊 *Pattern Type*: {entry_info['pattern']}
🕒 *Expected Confirmation*: {entry_info['confirm_time'].strftime('%Y-%m-%d %H:%M')}
📈 *Direction*: {'🟢 BUY' if entry_info['type'] == 'BUY' else '🔴 SELL'}

🎯 Potential Entry: {entry_info['potential_entry']}
🛑 Suggested Stop: {entry_info['stop_loss']}
💰 Initial Target: {entry_info['take_profit']}
📌 Confidence: {entry_info['confidence']}

📊 Key Levels:
   - Support: {entry_info['key_levels']['support'] if entry_info['key_levels']['support'] else 'N/A'}
   - Resistance: {entry_info['key_levels']['resistance'] if entry_info['key_levels']['resistance'] else 'N/A'}
   {f"- Fib 0.618: {entry_info['key_levels']['fib_618']:.4f}" if 'fib_618' in entry_info['key_levels'] else ''}

⚠️ *This is an EARLY WARNING* - Wait for confirmation at the expected time
"""
    return msg.strip()

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
    print(f"\n=== Starting new scan at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    
    for symbol in symbols_to_check:
        try:
            start_time = time.time()
            df = fetch_binance_data(symbol)
            
            if df is None or len(df) < 200:
                print(f"[{symbol}] Skipped - insufficient data")
                continue
                
            # Get regular signals
            result = predict_signal(df)
            
            # Get early entry predictions
            early_entries = predict_early_entry(df)
            
            processing_time = time.time() - start_time
            
            print(f"[{symbol}] {result['signal']} ({processing_time:.2f}s) - {result.get('confidence_score', '')}")
            
            # Send early entry alerts if they're new
            for entry in early_entries['potential_entries']:
                entry_key = f"{symbol}_{entry['pattern']}_{entry['confirm_time'].timestamp()}"
                
                if entry_key not in early_warnings:
                    msg = format_early_entry_message(symbol, entry)
                    send_telegram_message(msg)
                    early_warnings[entry_key] = entry
                    print(f"[EARLY ENTRY] {symbol} {entry['type']} potential detected")
            
            # Send regular signals if they're strong and new
            if result['signal'] in ["BUY", "SELL"]:
                if symbol not in last_alerts or last_alerts[symbol]['signal'] != result['signal']:
                    msg = format_message(symbol, result)
                    send_telegram_message(msg)
                    last_alerts[symbol] = result
                    
            time.sleep(0.5)  # Rate limit
            
        except Exception as e:
            print(f"[{symbol}] Error: {str(e)}")
            time.sleep(5)
    
    print(f"\n=== Completed full scan at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    print("Waiting for next scan in 30 minutes...")
    time.sleep(60 * 30)  # Wait 30 minutes between full scans
