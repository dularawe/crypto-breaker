import pandas as pd
import time
import requests
from binance.client import Client
from binance import ThreadedWebsocketManager
from ta.trend import SMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from datetime import datetime

# === Config ===
TELEGRAM_TOKEN = '7532851212:AAHOVx1esNWrtk2SJbBQHCXMae7Y-dKJR5o'
CHAT_ID = '7771111812'
TIMEFRAME = '1h'
INTERVAL_SECONDS = 600  # Every 30 minutes

symbols_to_check = [
    "FETUSDT",      
    "BIDUSDT",      
    "ETCUSDT",      
    "JTOUSDT",      
    "ONDOUSDT",     
    "JUPUSDT",      
    "SKATEUSDT",    
    "NXPCUSDT",     
    "1000BONKUSDT", 
    "TONUSDT",      
    "POPCATUSDT",
    "ICPUSDT",
    "PEOPLEUSDT",
    "AI16ZUSDT",
    "RPLUSDT",
    "AIXBTUSDT",
    "XLMUSDT",
    "KAITOUSDT",
    "1000FLOKIUSDT",
    "AUSDT",
    "COOKIEUSDT",
    "DEXEUSDT",
    "OMUSDT",
    "EPTUSDT",
    "APTUSDT",
    "ORDIUSDT",
    "GALAUSDT",
    "HBARUSDT",
    "SSVUSDT",
    "LQTYUSDT",
    "PENDLEUSDT",
    "BUSDT",
    "DYDXUSDT",
    "CAKEUSDT",
    "BOMEUSDT",
    "GOATUSDT",
    "SUSDT",
    "MOVEUSDT",
    "SUSHIUSDT",
    "SEIUSDT",
    "ATOMUSDT",
    "USUALUSDT",
    "RENDERUSDT",
    "MUBARAKUSDT",
    "1000000MOGUSDT",
    "APEUSDT",
    "ICXUSDT",
    "ZROUSDT",
    "HMSTRUSDT",
    "FLMUSDT",
    "TURBOUSDT",
    "VVVUSDT",
    "CHILLGUYUSDT",
    "RAYSOLUSDT",
    "ZENUSDT",
    "AXSUSDT",
    "MOVRUSDT",
    "BRETTUSDT",
    "SANDUSDT",
    "GRASSUSDT",
    "ALGOUSDT",
    "ARKMUSDT",
    "POLUSDT",
    "FIDAUSDT",
    "SOONUSDT",
    "GRIFFAINUSDT",
    "LAYERUSDT",
    "LISTAUSDT",
    "KAVAUSDT",
    "STRKUSDT",
    "XMRUSDT",
    "RUNEUSDT",
    "STXUSDT",
    "KMNOUSDT",
    "BERAUSDT",
    "SXTUSDT",
    "IMXUSDT",
    "VETUSDT",
    "ARCUSDT",
    "MEUSDT",
    "ARUSDT",
    "CETUSUSDT",
    "PYTHUSDT",
    "COWUSDT",
    "XAIUSDT",
    "PUMPUSDT",
    "VANAUSDT",
    "ZEREBROUSDT",
    "WUSDT",
    "1000SATSUSDT",
    "MEMEUSDT",
    "ALTUSDT",
    "SOLVUSDT",
    "IOSTUSDT",
    "HAEDALUSDT",
    "FXSUSDT",
    "NOTUSDT",
    "TUSDT",
    "ZKUSDT",
    "SHELLUSDT",
    "MEWUSDT",
    "KASUSDT",
    "SWARMSUSDT",
    "AEROUSDT",
    "JELLYJELLYUSDT",
    "AVAAIUSDT",
    "NEOUSDT",
    "THETAUSDT",
    "GRTUSDT",
    "JASMYUSDT",
    "KERNELUSDT",
    "B2USDT",
    "BDXNUSDT",
    "AERGOUSDT",
    "ACTUSDT",
    "GPSUSDT",
    "SNXUSDT",
    "DOGSUSDT",
    "PARTIUSDT",
    "GUNUSDT",
    "REZUSDT",
    "CVCUSDT",
    "NEIROETHUSDT",
    "VINEUSDT",
    "XTZUSDT",
    "IOUSDT",
    "BABYUSDT",
    "NILUSDT",
    "BANANAUSDT",
    "PONKEUSDT",
    "FUNUSDT",
    "SAGAUSDT",
    "SIRENUSDT",
    "GMXUSDT",
    "CFXUSDT",
    "SXPUSDT",
    "QNTUSDT",
    "MELANIAUSDT",
    "TSTUSDT",
    "MANAUSDT",
    "BIOUSDT",
    "ALCHUSDT",
    "GMTUSDT",
    "BMTUSDT",
    "DEEPUSDT",
    "MINAUSDT",
    "SIGNUSDT",
    "TUTUSDT",
    "FLOWUSDT",
    "MANTAUSDT",
    "ROSEUSDT",
    "CHESSUSDT",
    "CHZUSDT",
    "PROMPTUSDT",
    "DEGOUSDT",
    "ZRXUSDT",
    "ACHUSDT",
    "1INCHUSDT",
    "PLUMEUSDT",
    "AGLDUSDT",
    "CELOUSDT",
    "RSRUSDT",
    "IOTAUSDT",
    "ORCAUSDT",
    "API3USDT",
    "PIXELUSDT",
    "FORMUSDT",
    "BANANAS31USDT",
    "1MBABYDOGEUSDT",
    "IPUSDT",
    "PORTALUSDT",
    "WALUSDT",
    "DEGENUSDT",
    "HYPERUSDT",
    "MERLUSDT",
    "SUPERUSDT",
    "RAREUSDT",
    "MAGICUSDT",
    "DOLOUSDT",
    "PIPPINUSDT",
    "BBUSDT",
    "CYBERUSDT",
    "BLURUSDT",
    "EPICUSDT",
    "VANRYUSDT",
    "1000LUNCUSDT",
    "BROCCOLIF3BUSDT",
    "STOUSDT",
    "IOTXUSDT",
    "ZECUSDT",
    "BSVUSDT",
    "USDCUSDT",
    "AEVOUSDT",
    "EGLDUSDT",
    "ZILUSDT",
    "RLCUSDT",
    "AUCTIONUSDT",
    "ALICEUSDT",
    "BIGTIMEUSDT",
    "PORT3USDT",
    "HIPPOUSDT",
    "ONEUSDT",
    "ALPHAUSDT",
    "CATIUSDT",
    "METISUSDT",
    "LRCUSDT",
    "ATHUSDT",
    "BROCCOLI714USDT",
    "KSMUSDT",
    "KDAUSDT",
    "NMRUSDT",
    "STORJUSDT",
    "THEUSDT",
    "MORPHOUSDT",
    "MOCAUSDT",
    "SCRUSDT",
    "PUFFERUSDT",
    "QTUMUSDT",
    "XCNUSDT",
    "UXLINKUSDT",
    "CGPTUSDT",
    "GASUSDT",
    "AGTUSDT",
    "BELUSDT",
    "1000RATSUSDT",
    "VOXELUSDT",
    "DYMUSDT",
    "SKYAIUSDT",
    "DRIFTUSDT",
    "ENJUSDT",
    "1000CATUSDT",
    "LUNA2USDT",
    "WOOUSDT",
    "HOOKUSDT",
    "BAKEUSDT",
    "COTIUSDT",
    "OMNIUSDT",
    "YGGUSDT",
    "PUNDIXUSDT",
    "AKTUSDT",
    "SWELLUSDT",
    "ASTRUSDT",
    "EDUUSDT",
    "BEAMXUSDT",
    "HOTUSDT",
    "C98USDT",
    "HIGHUSDT",
    "JOEUSDT",
    "ZETAUSDT",
    "DASHUSDT",
    "OGNUSDT",
    "ANKRUSDT",
    "LUMIAUSDT",
    "ONTUSDT",
    "IDUSDT",
    "MYROUSDT",
    "CELRUSDT",
    "1000CHEEMSUSDT",
    "PHBUSDT",
    "HIVEUSDT",
    "FHEUSDT",
    "SONICUSDT",
    "REDUSDT",
    "XVGUSDT",
    "ATAUSDT",
    "AIUSDT",
    "REIUSDT",
    "SPELLUSDT",
    "AIOTUSDT",
    "HIFIUSDT",
    "BICOUSDT",
    "CKBUSDT",
    "OBOLUSDT",
    "CHRUSDT",
    "MAVUSDT",
    "STGUSDT",
    "PHAUSDT",
    "SKLUSDT",
    "SAFEUSDT",
    "DFUSDT",
    "AVAUSDT",
    "NFPUSDT",
    "HFTUSDT",
    "1000XUSDT",
    "1000XECUSDT",
    "GTCUSDT",
    "ALPINEUSDT",
    "BATUSDT",
    "AWEUSDT",
    "RDNTUSDT",
    "NKNUSDT",
    "DOODUSDT",
    "LEVERUSDT",
    "BANDUSDT",
    "SYNUSDT",
    "ARKUSDT",
    "KNCUSDT",
    "BSWUSDT",
    "ETHWUSDT",
    "JSTUSDT",
    "PERPUSDT",
    "ACXUSDT",
    "BANUSDT",
    "DENTUSDT",
    "GLMUSDT",
    "CTSIUSDT",
    "RONINUSDT",
    "MEMEFIUSDT",
    "LOKAUSDT",
    "TOKENUSDT",
    "ZKJUSDT",
    "TRUUSDT",
    "B3USDT",
    "SUNUSDT",
    "FISUSDT",
    "POLYXUSDT",
    "MILKUSDT",
    "HEIUSDT",
    "TNSRUSDT",
    "TWTUSDT",
    "KOMAUSDT",
    "VTHOUSDT",
    "RIFUSDT",
    "MLNUSDT",
    "TLMUSDT",
    "WAXPUSDT",
    "OGUSDT",
    "USTCUSDT",
    "ACEUSDT",
    "MBOXUSDT",
    "LSKUSDT",
    "OXTUSDT",
    "ASRUSDT",
    "BANKUSDT",
    "ARPAUSDT",
    "SFPUSDT",
    "ILVUSDT",
    "NTRNUSDT",
    "DUSKUSDT",
    "MTLUSDT",
    "VELODROMEUSDT",
    "GUSDT",
    "POWRUSDT",
    "DIAUSDT",
    "FLUXUSDT",
    "XVSUSDT",
    "BNTUSDT",
    "DODOXUSDT",
    "CTKUSDT",
    "PROMUSDT",
    "FIOUSDT",
    "SYSUSDT",
    "BRUSDT",
    "SCRTUSDT",
    "QUICKUSDT",
    "COSUSDT",
    "STEEMUSDT",
    "DUSDT",
    "MAVIAUSDT",
    "1000WHYUSDT",
    "FORTHUSDT",
    "ONGUSDT",
    "GHSTUSDT",
    "SANTOSUSDT",
    "VICUSDT"
    "FLUXUSDT",
    "XVSUSDT",
    "BNTUSDT",
    "DODOXUSDT",
    "CTKUSDT",
    "PROMUSDT",
    "FIOUSDT",
    "SYSUSDT",
    "BRUSDT",
    "SCRTUSDT",
    "QUICKUSDT",
    "COSUSDT",
    "STEEMUSDT",
    "DUSDT",
    "MAVIAUSDT",
    "1000WHYUSDT",
    "FORTHUSDT",
    "ONGUSDT",
    "GHSTUSDT",
    "SANTOSUSDT",
    "VICUSDT"
    "SCRTUSDT",
    "QUICKUSDT",
    "COSUSDT",
    "STEEMUSDT",
    "DUSDT",
    "MAVIAUSDT",
    "1000WHYUSDT",
    "FORTHUSDT",
    "ONGUSDT",
    "GHSTUSDT",
    "SANTOSUSDT",
    "VICUSDT"
    "VICUSDT"
]

# Global variables
client = Client()
twm = ThreadedWebsocketManager()
twm.start()
live_data = {symbol: {'price': None, 'volume': None} for symbol in symbols_to_check}

def get_klines(symbol, interval=TIMEFRAME, limit=150):
    try:
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        df = pd.DataFrame(klines, columns=[
            'time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'num_trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        df['close'] = df['close'].astype(float)
        df['low'] = df['low'].astype(float)
        df['high'] = df['high'].astype(float)
        df['volume'] = df['volume'].astype(float)
        return df
    except Exception as e:
        print(f"❌ Error fetching {symbol}: {e}")
        return None

def handle_socket_message(msg):
    """Handle incoming WebSocket messages"""
    try:
        symbol = msg['s']
        if symbol in live_data:
            live_data[symbol]['price'] = float(msg['c'])
            live_data[symbol]['volume'] = float(msg['v'])
    except Exception as e:
        print(f"WebSocket error: {e}")

def start_websocket():
    """Start WebSocket connections for all symbols"""
    for symbol in symbols_to_check:
        twm.start_symbol_ticker_socket(
            callback=handle_socket_message,
            symbol=symbol
        )
    print("WebSocket connections established for all symbols")

def detect_breakout_method(df, symbol):
    df['MA7'] = SMAIndicator(df['close'], window=7).sma_indicator()
    df['MA25'] = SMAIndicator(df['close'], window=25).sma_indicator()
    df['MA99'] = SMAIndicator(df['close'], window=99).sma_indicator()
    df['RSI'] = RSIIndicator(df['close'], window=14).rsi()
    bb = BollingerBands(close=df['close'], window=20)
    df['bb_upper'] = bb.bollinger_hband()

    last = df.iloc[-1]
    prev = df.iloc[-2]
    resistance = df['close'][-25:-4].max()
    
    # Use live price if available, otherwise use last close
    current_price = live_data[symbol]['price'] or last['close']
    current_volume = live_data[symbol]['volume'] or last['volume']
    
    entry_price = current_price
    entry_time = datetime.now().strftime('%Y-%m-%d %H:%M')

    # ✅ Accumulation zone near horizontal resistance
    price_was_under_resistance = prev['close'] < resistance * 0.995
    price_now_above = entry_price > resistance
    accumulation_range_ok = df['close'][-25:-4].max() - df['close'][-25:-4].min() < resistance * 0.08

    # ✅ MA compression breakout logic
    ma_compression = abs(last['MA7'] - last['MA25']) < resistance * 0.005 and \
                     abs(last['MA25'] - last['MA99']) < resistance * 0.01
    ma_cross_up = prev['MA7'] < prev['MA25'] and last['MA7'] > last['MA25']
    rsi_ok = 45 < last['RSI'] < 70
    bb_touch = entry_price > prev['bb_upper']
    volume_spike = current_volume > df['volume'].mean() * 1.2

    # Confirm all
    conditions = [
        price_was_under_resistance, price_now_above,
        ma_cross_up, ma_compression, rsi_ok,
        bb_touch, volume_spike, accumulation_range_ok
    ]
    if all(conditions):
        return True, entry_price, entry_time, resistance
    else:
        return False, None, None, None

def format_signal(symbol, price, time_str, resistance):
    tp1 = round(price * 1.01, 5)
    tp2 = round(price * 1.02, 5)
    tp3 = round(price * 1.03, 5)
    sl = round(resistance * 0.995, 5)
    return f"""🚀 *MA Compression Breakout Detected!*

🪙 *Symbol:* `{symbol}`
📍 *Entry Price:* `{price}`
🕒 *Entry Time:* `{time_str}`
📊 *Pattern:* MA Compression + Volume Break + BB Spike

🎯 *Take Profit:*
• TP1 = {tp1}
• TP2 = {tp2}
• TP3 = {tp3}

🛡 *Stop Loss:* {sl}
⏰ *Timeframe:* {TIMEFRAME.upper()}
"""

def send_telegram(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        'chat_id': CHAT_ID,
        'text': msg,
        'parse_mode': 'Markdown'
    }
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print("Telegram Error:", e)

def scan_all():
    print("📡 Scanning 1H MA Breakouts...\n")
    for i, symbol in enumerate(symbols_to_check, 1):
        print(f"🔍 Checking {symbol} ({i}/{len(symbols_to_check)})...")
        df = get_klines(symbol)
        if df is None:
            continue

        valid, price, time_str, resistance = detect_breakout_method(df, symbol)
        if valid:
            msg = format_signal(symbol, price, time_str, resistance)
            send_telegram(msg)
            print(f"✅ SIGNAL SENT: {symbol}")

def main():
    # Start WebSocket connections
    start_websocket()
    
    # Initial scan
    scan_all()
    
    # 🔁 Loop
    while True:
        print(f"\n⏱ Waiting {INTERVAL_SECONDS // 60} min for next scan...\n")
        time.sleep(INTERVAL_SECONDS)
        scan_all()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nShutting down...")
        twm.stop()
    except Exception as e:
        print(f"Unexpected error: {e}")
        twm.stop()