import pandas as pd
import time
import requests
import logging
from binance.client import Client
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from ta.volume import VolumeWeightedAveragePrice
from datetime import datetime, timedelta

# === Configuration ===
TELEGRAM_TOKEN = '7532851212:AAHOVx1esNWrtk2SJbBQHCXMae7Y-dKJR5o'
CHAT_ID = '7771111812'
TIMEFRAME = '1h'
INTERVAL_MINUTES = 10
MAX_RETRIES = 5
BASE_RECONNECT_DELAY = 1
SYMBOL_REFRESH_HOURS = 6

DEFAULT_SYMBOLS =  [
    "LAUSDT",       
    "ONDOUSDT",     
    "FETUSDT",      
    "UMAUSDT",
    "TONUSDT",
    "1000BONKUSDT",
    "SOPHUSDT",
    "POPCATUSDT",
    "SKATEUSDT",
    "NXPCUSDT",
    "PEOPLEUSDT",
    "ICPUSDT",
    "BIDUSDT",
    "XLMUSDT",
    "AI16ZUSDT",
    "AIXBTUSDT",
    "OMUSDT",
    "1000FLOKIUSDT",
    "ICXUSDT",
    "KAITOUSDT",
    "APTUSDT",
    "ORDIUSDT",
    "COOKIEUSDT",
    "GALAUSDT",
    "HBARUSDT",
    "SSVUSDT",
    "LQTYUSDT",
    "DEXEUSDT",
    "DYDXUSDT",
    "PENDLEUSDT",
    "GOATUSDT",
    "CAKEUSDT",
    "BUSDT",
    "MOVEUSDT",
    "BOMEUSDT",
    "SUSHIUSDT",
    "SUSDT",
    "USUALUSDT",
    "AXSUSDT",
    "SEIUSDT",
    "ATOMUSDT",
    "RENDERUSDT",
    "MUBARAKUSDT",
    "ZROUSDT",
    "APEUSDT",
    "TURBOUSDT",
    "FLMUSDT",
    "EPTUSDT",
    "RAYSOLUSDT",
    "CHILLGUYUSDT",
    "ZENUSDT",
    "VVVUSDT",
    "HMSTRUSDT",
    "1000000MOGUSDT",
    "KMNOUSDT",
    "SANDUSDT",
    "MOVRUSDT",
    "BRETTUSDT",
    "GRASSUSDT",
    "ARKMUSDT",
    "ALGOUSDT",
    "FIDAUSDT",
    "POLUSDT",
    "LAYERUSDT",
    "GRIFFAINUSDT",
    "LISTAUSDT",
    "STRKUSDT",
    "IOSTUSDT",
    "SOONUSDT",
    "XMRUSDT",
    "KAVAUSDT",
    "SXTUSDT",
    "RUNEUSDT",
    "STXUSDT",
    "IMXUSDT",
    "COWUSDT",
    "BERAUSDT",
    "ARCUSDT",
    "ARUSDT",
    "VETUSDT",
    "XAIUSDT",
    "PUMPUSDT",
    "WUSDT",
    "PYTHUSDT",
    "VANAUSDT",
    "NOTUSDT",
    "FXSUSDT",
    "CETUSUSDT",
    "ZEREBROUSDT",
    "SOLVUSDT",
    "1000SATSUSDT",
    "ALTUSDT",
    "MEMEUSDT",
    "HAEDALUSDT",
    "ZKUSDT",
    "DOGSUSDT",
    "SHELLUSDT",
    "MEWUSDT",
    "MEUSDT",
    "KASUSDT",
    "AEROUSDT",
    "SWARMSUSDT",
    "B2USDT",
    "NEOUSDT",
    "GRTUSDT",
    "THETAUSDT",
    "AVAAIUSDT",
    "SNXUSDT",
    "KERNELUSDT",
    "BDXNUSDT",
    "XTZUSDT",
    "REZUSDT",
    "NEIROETHUSDT",
    "PARTIUSDT",
    "BABYUSDT",
    "SIRENUSDT",
    "IOUSDT",
    "VINEUSDT",
    "ACTUSDT",
    "NILUSDT",
    "CYBERUSDT",
    "GUNUSDT",
    "BANANAUSDT",
    "CVCUSDT",
    "SXPUSDT",
    "MELANIAUSDT",
    "FUNUSDT",
    "GPSUSDT",
    "PONKEUSDT",
    "MANAUSDT",
    "SAGAUSDT",
    "JELLYJELLYUSDT",
    "TSTUSDT",
    "JASMYUSDT",
    "BMTUSDT",
    "FORMUSDT",
    "QNTUSDT",
    "GMXUSDT",
    "GMTUSDT",
    "MINAUSDT",
    "BIOUSDT",
    "MANTAUSDT",
    "ALCHUSDT",
    "SIGNUSDT",
    "FLOWUSDT",
    "AERGOUSDT",
    "DEEPUSDT",
    "ROSEUSDT",
    "CFXUSDT",
    "ZRXUSDT",
    "PLUMEUSDT",
    "CHZUSDT",
    "TUSDT",
    "ACHUSDT",
    "CHESSUSDT",
    "PROMPTUSDT",
    "1INCHUSDT",
    "TUTUSDT",
    "AGLDUSDT",
    "IOTAUSDT",
    "CELOUSDT",
    "RSRUSDT",
    "PORTALUSDT",
    "ORCAUSDT",
    "PIXELUSDT",
    "DEGOUSDT",
    "MERLUSDT",
    "1MBABYDOGEUSDT",
    "BANANAS31USDT",
    "WALUSDT",
    "STOUSDT",
    "EPICUSDT",
    "IPUSDT",
    "HYPERUSDT",
    "MAGICUSDT",
    "MOCAUSDT",
    "BLURUSDT",
    "AEVOUSDT",
    "DRIFTUSDT",
    "VANRYUSDT",
    "PIPPINUSDT",
    "BBUSDT",
    "API3USDT",
    "USDCUSDT",
    "BSVUSDT",
    "SUPERUSDT",
    "AUCTIONUSDT",
    "1000LUNCUSDT",
    "DEGENUSDT",
    "ZILUSDT",
    "BIGTIMEUSDT",
    "EGLDUSDT",
    "ZECUSDT",
    "BROCCOLIF3BUSDT",
    "RAREUSDT",
    "IOTXUSDT",
    "ONEUSDT",
    "HIPPOUSDT",
    "RLCUSDT",
    "ALICEUSDT",
    "MORPHOUSDT",
    "KSMUSDT",
    "CATIUSDT",
    "ALPHAUSDT",
    "LRCUSDT",
    "BROCCOLI714USDT",
    "THEUSDT",
    "CGPTUSDT",
    "STORJUSDT",
    "SCRUSDT",
    "AIUSDT",
    "NMRUSDT",
    "AGTUSDT",
    "ATHUSDT",
    "GASUSDT",
    "METISUSDT",
    "SKYAIUSDT",
    "1000RATSUSDT",
    "ENJUSDT",
    "PORT3USDT",
    "VOXELUSDT",
    "PUFFERUSDT",
    "BAKEUSDT",
    "QTUMUSDT",
    "XCNUSDT",
    "DOLOUSDT",
    "COTIUSDT",
    "DYMUSDT",
    "HIGHUSDT",
    "1000CATUSDT",
    "KDAUSDT",
    "LUNA2USDT",
    "OMNIUSDT",
    "C98USDT",
    "YGGUSDT",
    "SWELLUSDT",
    "WOOUSDT",
    "BELUSDT",
    "UXLINKUSDT",
    "DASHUSDT",
    "ASTRUSDT",
    "AKTUSDT",
    "EDUUSDT",
    "ZETAUSDT",
    "BEAMXUSDT",
    "HOTUSDT",
    "ONTUSDT",
    "OGNUSDT",
    "HOOKUSDT",
    "JOEUSDT",
    "PUNDIXUSDT",
    "FHEUSDT",
    "IDUSDT",
    "1000CHEEMSUSDT",
    "ANKRUSDT",
    "LUMIAUSDT",
    "PHBUSDT",
    "SONICUSDT",
    "XVGUSDT",
    "HIVEUSDT",
    "REDUSDT",
    "MYROUSDT",
    "CELRUSDT",
    "ATAUSDT",
    "REIUSDT",
    "SPELLUSDT",
    "HIFIUSDT",
    "SAFEUSDT",
    "HFTUSDT",
    "BICOUSDT",
    "CKBUSDT",
    "OBOLUSDT",
    "CHRUSDT",
    "SKLUSDT",
    "DFUSDT",
    "STGUSDT",
    "MAVUSDT",
    "NFPUSDT",
    "ACXUSDT",
    "AVAUSDT",
    "PHAUSDT",
    "DOODUSDT",
    "1000XUSDT",
    "AIOTUSDT",
    "BATUSDT",
    "GTCUSDT",
    "1000XECUSDT",
    "RDNTUSDT",
    "NKNUSDT",
    "KNCUSDT",
    "ALPINEUSDT",
    "LEVERUSDT",
    "PERPUSDT",
    "BANDUSDT",
    "SYNUSDT",
    "TRUUSDT",
    "AWEUSDT",
    "SUNUSDT",
    "ETHWUSDT",
    "BSWUSDT",
    "DENTUSDT",
    "JSTUSDT",
    "RONINUSDT",
    "GLMUSDT",
    "FISUSDT",
    "MEMEFIUSDT",
    "TOKENUSDT",
    "B3USDT",
    "LOKAUSDT",
    "ARKUSDT",
    "ACEUSDT",
    "MILKUSDT",
    "POLYXUSDT",
    "TNSRUSDT",
    "VTHOUSDT",
    "TWTUSDT",
    "MLNUSDT",
    "RIFUSDT",
    "HEIUSDT",
    "TLMUSDT",
    "ZKJUSDT",
    "CTSIUSDT",
    "KOMAUSDT",
    "ARPAUSDT",
    "WAXPUSDT",
    "ASRUSDT",
    "LSKUSDT",
    "USTCUSDT",
    "OXTUSDT",
    "SFPUSDT",
    "MBOXUSDT",
    "ILVUSDT",
    "BANKUSDT",
    "OGUSDT",
    "VELODROMEUSDT",
    "NTRNUSDT",
    "GUSDT",
    "DUSKUSDT",
    "PROMUSDT",
    "MTLUSDT",
    "POWRUSDT",
    "FLUXUSDT",
    "BNTUSDT",
    "DIAUSDT",
    "XVSUSDT",
    "DODOXUSDT",
    "BRUSDT",
    "VICUSDT",
    "SYSUSDT",
    "FIOUSDT",
    "CTKUSDT",
    "SCRTUSDT",
    "STEEMUSDT",
    "MAVIAUSDT",
    "QUICKUSDT",
    "DUSDT",
    "ONGUSDT",
    "FORTHUSDT",
    "BANUSDT",
    "COSUSDT",
    "GHSTUSDT"
]

# === Logging Setup ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('breakout_scanner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Binance client
client = Client()

def get_current_price(symbol):
    try:
        ticker = client.get_symbol_ticker(symbol=symbol)
        return float(ticker['price'])
    except Exception as e:
        logger.error(f"Error getting current price for {symbol}: {e}")
        return None

def get_active_symbols():
    try:
        all_symbols = []
        exchange_info = client.get_exchange_info()
        for symbol in exchange_info['symbols']:
            if symbol['quoteAsset'] == 'USDT' and symbol['status'] == 'TRADING':
                all_symbols.append(symbol['symbol'])
        return all_symbols[:50]
    except Exception as e:
        logger.error(f"Error in get_active_symbols: {e}")
        return DEFAULT_SYMBOLS

def get_klines(symbol, interval=TIMEFRAME, limit=100):
    try:
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        df = pd.DataFrame(klines, columns=[
            'time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'num_trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_cols] = df[numeric_cols].astype(float)
        return df
    except Exception as e:
        logger.warning(f"Failed to get klines for {symbol}: {e}")
        return None

def detect_breakout(df, symbol):
    try:
        df['MA7'] = EMAIndicator(df['close'], window=7).ema_indicator()
        df['MA25'] = EMAIndicator(df['close'], window=25).ema_indicator()
        df['MA99'] = EMAIndicator(df['close'], window=99).ema_indicator()
        df['RSI14'] = RSIIndicator(df['close'], window=14).rsi()
        df['VOL_MA20'] = df['volume'].rolling(20).mean()

        resistance_window = df['high'][-50:-5]
        resistance = resistance_window.max()
        resistance_touches = sum(abs(resistance_window - resistance) < resistance * 0.005)

        current_price = get_current_price(symbol)
        if current_price is None:
            return False, None, None, None, None, None, None, None, None

        last = df.iloc[-1]

        conditions = {
            'resistance_break': current_price > resistance * 1.005,
            'multiple_tests': resistance_touches >= 3,
            'bullish_candle': last['close'] > last['open'],
            'volume_spike': last['volume'] > df['VOL_MA20'].iloc[-1] * 1.5,
            'ma_alignment': last['MA7'] > last['MA25'] > last['MA99'],
            'above_vwap': current_price > VolumeWeightedAveragePrice(
                high=df['high'], low=df['low'], close=df['close'],
                volume=df['volume'], window=20).volume_weighted_average_price().iloc[-1]
        }

        if all(conditions.values()):
            measured_move = resistance * 0.07
            tp1 = round(resistance + measured_move * 0.5, 4)
            tp2 = round(resistance + measured_move, 4)
            tp3 = round(resistance + measured_move * 1.5, 4)
            sl = round(resistance * 0.995, 4)

            return True, current_price, datetime.now().strftime('%Y-%m-%d %H:%M'), resistance, tp1, tp2, tp3, sl, resistance_touches
    except Exception as e:
        logger.error(f"Breakout detection error for {symbol}: {e}")

    return False, None, None, None, None, None, None, None, None

def format_breakout_alert(symbol, price, time_str, resistance, tp1, tp2, tp3, sl, resistance_touches):
    return f"""🚀 *Breakout Alert* ({TIMEFRAME.upper()})

📈 *Symbol:* `{symbol}`
💰 *Price:* `{price}`
⏰ *Time:* `{time_str}`
📊 *Resistance:* `{resistance}` (Tested {resistance_touches}x)

🎯 *Targets (Measured Move):*
├ TP1: {tp1} (+{round((tp1/resistance-1)*100, 2)}%)
├ TP2: {tp2} (+{round((tp2/resistance-1)*100, 2)}%)
└ TP3: {tp3} (+{round((tp3/resistance-1)*100, 2)}%)

🛡 *Stop Loss:* {sl} (-{round((1-sl/price)*100, 2)}%)

📊 *Technical Analysis Summary:*
1. *Breakout Pattern:* Resistance at `{round(resistance, 4)}` broken with strong volume.
2. *Breakout Candle:* Large bullish candle above resistance confirms move.
3. *Measured Move:* Estimated target zone `{round(resistance * 1.07, 4)}` (+7%) from breakout.
4. *Volume Confirmation:* Volume spike detected (1.5x above average).
5. *MA Alignment:* MA(7) > MA(25) > MA(99) → Bullish trend confirmation.

📉 *Indicators Used:*
- MA(7): Yellow
- MA(25): Pink
- MA(99): Purple
- RSI(14)
- VWAP(20)
- Volume Bars

⏳ *Next Scan in {INTERVAL_MINUTES} minutes*
"""

def send_telegram_alert(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        'chat_id': CHAT_ID,
        'text': msg,
        'parse_mode': 'Markdown'
    }
    try:
        response = requests.post(url, data=payload, timeout=10)
        response.raise_for_status()
        return True
    except Exception as e:
        logger.error(f"Failed to send Telegram alert: {e}")
        return False

def scan_symbols(symbols):
    signals_found = 0
    for symbol in symbols:
        df = get_klines(symbol)
        if df is None or len(df) < 50:
            continue
        valid, price, time_str, res, tp1, tp2, tp3, sl, resistance_touches = detect_breakout(df, symbol)
        if valid:
            msg = format_breakout_alert(symbol, price, time_str, res, tp1, tp2, tp3, sl, resistance_touches)
            if send_telegram_alert(msg):
                signals_found += 1
                logger.info(f"Breakout alert sent for {symbol}")
    return signals_found

def run_continuous_scanner():
    last_symbol_refresh = datetime.now()
    symbols = get_active_symbols()

    while True:
        try:
            if (datetime.now() - last_symbol_refresh) > timedelta(hours=SYMBOL_REFRESH_HOURS):
                symbols = get_active_symbols()
                last_symbol_refresh = datetime.now()
                logger.info(f"Refreshed symbol list: {len(symbols)} symbols")

            start_time = datetime.now()
            logger.info(f"Starting scan at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            found = scan_symbols(symbols)
            logger.info(f"Scan complete. {found} breakouts found.")

            elapsed = (datetime.now() - start_time).total_seconds()
            time.sleep(max(0, INTERVAL_MINUTES * 60 - elapsed))
        except KeyboardInterrupt:
            logger.info("Scanner stopped by user.")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    run_continuous_scanner()
