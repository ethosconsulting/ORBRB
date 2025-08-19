#24 ORBRB
import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mdates
import pytz
from datetime import datetime, timedelta, time
import numpy as np
import seaborn as sns
import os
from itertools import product
from typing import Dict, Optional, Tuple, Any

# Streamlit app configuration
st.set_page_config(layout="wide", page_title="Trading Strategy Analyzer")

# Main title
st.title("Trading Strategy Analyzer")
st.markdown("""
This app analyzes a breakout trading strategy based on the opening range of the trading day.
""")

# Sidebar for user inputs
with st.sidebar:
    st.header("Strategy Parameters")

    # Ticker symbol
    ticker = st.text_input("Ticker Symbol", value='ES=F')

    # Timezone selection
    timezone = st.selectbox(
        "Timezone",
        ['Europe/London', 'America/New_York', 'Asia/Tokyo'],
        index=0
    )

    data_source = st.selectbox(
        "Select Data Source",
        ["YFinance", "Upload CSV"],
        index=0,
        help="Choose between downloading from Yahoo Finance or uploading your own CSV file"
    )

    uploaded_file = None
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader(
            "Upload CSV File",
            type=["csv"],
            help="Upload a CSV file in the same format as the downloaded data"
        )

    # Trading session hours - using string options but converting to int later
    st.subheader("Trading Session Hours")
    col1, col2 = st.columns(2)
    with col1:
        start_hour = st.number_input("Start Hour (0-23)", min_value=0, max_value=23, value=12)
        start_minute = st.selectbox("Start Minute", options=['00', '15', '30', '45', '59'], index=0)
    with col2:
        end_hour = st.number_input("End Hour (0-23)", min_value=0, max_value=23, value=18)
        end_minute = st.selectbox("End Minute", options=['00', '15', '30', '45','59'], index=0)

    # Opening range configuration
    st.subheader("Opening Range Configuration")
    col1, col2 = st.columns(2)
    with col1:
        or_start_hour = st.number_input("OR Start Hour (0-23)", min_value=0, max_value=23, value=14)
        or_start_minute = st.selectbox("OR Start Minute",
                                    options=['00', '05', '10', '15', '20', '25', '30', '35', '40', '45', '50', '55'],
                                    index=6)  # Default to '30'
    with col2:
        or_end_hour = st.number_input("OR End Hour (0-23)", min_value=0, max_value=23, value=14)
        or_end_minute = st.selectbox("OR End Minute",
                                    options=['00', '05', '10', '15', '20', '25', '30', '35', '40', '45', '50', '55'],
                                    index=11)  # Default to '55' for '30' min close candle for NYC session.

    # Date range
    st.subheader("Date Range")
    col1, col2 = st.columns(2)
    with col1:
        # Set start_date to 50 days before today
        start_date = st.date_input("Start Date",
                                  value=datetime.now() - timedelta(days=55),
                                  max_value=datetime.now() - timedelta(days=1))
    with col2:
        # Set end_date to today
        end_date = st.date_input("End Date",
                                value=datetime.now(),
                                max_value=datetime.now())

    st.subheader("TP-SL Range (Absolute Points)")
    col1, col2 = st.columns(2)
    with col1:
        tp_min = st.number_input("TP Min (points)", min_value=0.1, max_value=100.0, value=4.0, step=0.5)
        tp_max = st.number_input("TP Max (points)", min_value=0.1, max_value=100.0, value=10.0, step=0.5)
        tp_step = st.number_input("TP Step", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    with col2:
        sl_min = st.number_input("SL Min (points)", min_value=0.1, max_value=100.0, value=4.0, step=0.5)
        sl_max = st.number_input("SL Max (points)", min_value=0.1, max_value=100.0, value=10.0, step=0.5)
        sl_step = st.number_input("SL Step", min_value=0.1, max_value=10.0, value=1.0, step=0.1)

    st.subheader("Optimization Settings")
    skip_poor_rr = st.checkbox("Skip poor risk-reward (SL < TP)", value=True)


    # Input: Friction parameters
    st.subheader("Friction parameters")
    buffer_pts = st.sidebar.number_input("Buffer (pts)", min_value=0.0, max_value=10.0, value=0.25, step=0.05)
    cost_pts = st.sidebar.number_input("Commission + Slippage (pts)", min_value=0.0, max_value=10.0, value=0.5, step=0.1)

    # Breakout Validation Parameters
    st.subheader("Breakout Validation Parameters")

    breakout_requires_close_outside = st.checkbox("Breakout requires close outside OR", value=True)

    col1, col2 = st.columns(2)
    with col1:
        min_breakout_pct = st.number_input(
            "Min Breakout % of OR",
            min_value=0.0, max_value=100.0, value=0.0, step=1.0
        ) / 100  # Convert % to decimal
    with col2:
        max_breakout_pct = st.number_input(
            "Max Breakout % of OR",
            min_value=0.0, max_value=100.0, value=100.0, step=1.0
        ) / 100  # Convert % to decimal

    min_volume_pct = st.number_input(
            "Min Valid Breakout Volume % of EMA21",
            min_value=0.0, max_value=500.0, value=90.0, step=5.0
        ) / 100  # Convert % to decimal

    # Reversal Settings
    st.subheader("Reversal Settings")
    max_reversal_bars = st.number_input(
        "Max Bars for Valid Reversal",
        min_value=1,
        max_value=24,  # You can adjust max as needed
        value=1,
        help="Number of candles allowed after breakout to find reversal (1=strict immediate)"
    )

    st.header("Strategy Type")
    strategy_type = st.selectbox(
        "Select Strategy Type",
        ["ORB", "RB", "Both"],
        index=0,
        help="ORB: Opening Range Breakout | RB: Rubber Band Reversal"
    )

# Helper functions (copied from your code with minor modifications)
def safe_float(value) -> float:
    """Type-safe conversion for pandas/float inputs"""
    if isinstance(value, (pd.Series, pd.DataFrame)):
        return float(value.iloc[0])
    return float(value)

def get_trading_days_data(start_date, end_date, ticker, timezone) -> pd.DataFrame:
    """Fetch 5-minute data for the specified date range"""
    # Convert string dates to datetime objects
    start_dt = pd.to_datetime(start_date).tz_localize(timezone)
    end_dt = pd.to_datetime(end_date).tz_localize(timezone) + timedelta(days=1)  # Include full end date

    # Download data with buffer days
    data = yf.download(
        tickers=ticker,
        interval='5m',
        start=start_dt - timedelta(days=2),  # Buffer for timezone issues
        end=end_dt + timedelta(days=2),      # Buffer for timezone issues
        progress=False
    )

    # Convert to specified timezone
    if data.index.tz is None:
        data.index = data.index.tz_localize('UTC')
    data.index = data.index.tz_convert(timezone)

    # Filter for the exact date range
    data = data[(data.index >= start_dt) & (data.index <= end_dt)]

    return data

def split_data_by_day(data: pd.DataFrame, start_hour, start_minute, end_hour, end_minute) -> Dict[datetime.date, pd.DataFrame]:
    """Split the data into separate days"""
    daily_data = {}
    for date, group in data.groupby(data.index.date):
        # Filter for trading hours
        trading_day = group[
            (group.index.time >= pd.to_datetime(f'{start_hour}:{start_minute}').time()) &
            (group.index.time <= pd.to_datetime(f'{end_hour}:{end_minute}').time())
        ]
        if not trading_day.empty:
            daily_data[date] = trading_day
    return daily_data

def get_opening_range_levels(data: pd.DataFrame, or_start_hour: int, or_start_minute: str,
                           or_end_hour: int, or_end_minute: str,
                           start_hour: int, start_minute: str,
                           end_hour: int, end_minute: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[int]]:
    """Returns (high, low, atr, expected_candles) with proper type handling"""
    # Convert string minutes to integers
    or_start_min = int(or_start_minute)
    or_end_min = int(or_end_minute)

    start_time = time(or_start_hour, or_start_min)
    end_time = time(or_end_hour, or_end_min)

    mask = (data.index.time >= start_time) & (data.index.time <= end_time)
    opening_range = data.loc[mask]

    if opening_range.empty:
        return None, None, None, None

    # Calculate expected candles for the full session
    session_start = time(start_hour, int(start_minute))
    session_end = time(end_hour, int(end_minute))
    session_date = data.index[0].date()
    session_duration = (datetime.combine(session_date, session_end) -
                      datetime.combine(session_date, session_start)).total_seconds() / 60
    expected_candles = int(session_duration // 5)

    high = safe_float(opening_range['High'].max())
    low = safe_float(opening_range['Low'].min())
    close = safe_float(opening_range['Close'].iloc[-1])

    tr = max(
        high - low,
        abs(high - close),
        abs(low - close)
    )

    return high, low, tr, expected_candles

def optimize_tp_sl(daily_data: Dict[datetime.date, pd.DataFrame], tp_range, sl_range, buffer: float, cost: float) -> pd.DataFrame:
    """Grid search optimization using absolute TP/SL values"""
    results = []

    for tp_val, sl_val in product(tp_range, sl_range):
        # Skip poor risk-reward combinations
        if skip_poor_rr and tp_val < sl_val:
            continue

        metrics = {
            'tp_value': tp_val,
            'sl_value': sl_val,
            'total_pnl': 0,
            'win_rate': 0,
            'trades': 0,
            'avg_pnl': 0
        }

        for date, day_data in daily_data.items():
            high, low, tr, expected_candles = get_opening_range_levels(
                day_data, or_start_hour, or_start_minute, or_end_hour, or_end_minute,
                start_hour, start_minute, end_hour, end_minute
            )
            if None in (high, low, tr):
                continue

            trade = simulate_trade(
                data=day_data,
                opening_high=high,
                opening_low=low,
                atr=tr,  # Still calculated but not used for TP/SL
                expected_candles=expected_candles,
                tp_value=tp_val,  # Now using absolute value
                sl_value=sl_val,  # Now using absolute value
                or_start_hour=or_start_hour,
                or_start_minute=or_start_minute,
                or_end_hour=or_end_hour,
                or_end_minute=or_end_minute,
                buffer=buffer,
                cost=cost,
                max_reversal_bars=max_reversal_bars,
                min_breakout_pct=min_breakout_pct,
                max_breakout_pct=max_breakout_pct,
                min_volume_pct=min_volume_pct
            )

            if trade:
                metrics['total_pnl'] += trade['pnl']
                metrics['win_rate'] += 1 if trade['pnl'] > 0 else 0
                metrics['trades'] += 1

        if metrics['trades'] > 0:
            metrics['win_rate'] = metrics['win_rate'] / metrics['trades'] * 100
            metrics['avg_pnl'] = metrics['total_pnl'] / metrics['trades']
            results.append(metrics)

    return pd.DataFrame(results).sort_values('total_pnl', ascending=False)

def check_breakout(
    current_idx: pd.Timestamp,
    current_low: float,
    current_high: float,
    current_open: float,
    current_close: float,
    current_volume: float,
    current_volume_ema21: float,
    opening_high: float,
    opening_low: float,
    min_breakout_pct: float,
    max_breakout_pct: float,
    min_volume_pct: float,
    opening_range_end_time: time,
    last_breakout_time: Optional[pd.Timestamp] = None
) -> Tuple[bool, Optional[str], float, float, pd.Timestamp]:
    """
    Returns:
        breakout_occurred (bool),
        direction (long/short/None),
        breakout_distance (float),
        breakout_pct_of_or (float),
        breakout_time (pd.Timestamp)
    """
    # Skip if we already have a breakout today
    if last_breakout_time is not None and current_idx.date() == last_breakout_time.date():
        return False, None, 0, 0, current_idx

    # Skip if before opening range end time
    if current_idx.time() <= opening_range_end_time:
        return False, None, 0, 0, current_idx

    or_range = opening_high - opening_low
    volume_pct = current_volume / current_volume_ema21 if current_volume_ema21 > 0 else 0
    breakout_occurred = False
    breakout_direction = None
    breakout_distance = 0
    breakout_pct_of_or = 0

    if breakout_requires_close_outside:
        if (current_close > opening_high) and (current_open < opening_high):  # Long breakout
            breakout_distance = current_close - opening_high
            breakout_pct_of_or = breakout_distance / or_range
            if (min_breakout_pct <= breakout_pct_of_or <= max_breakout_pct) and (volume_pct >= min_volume_pct):
                breakout_occurred = True
                breakout_direction = 'long'

        elif (current_close < opening_low) and (current_open > opening_low):  # Short breakout
            breakout_distance = opening_low - current_close
            breakout_pct_of_or = breakout_distance / or_range
            if (min_breakout_pct <= breakout_pct_of_or <= max_breakout_pct) and (volume_pct >= min_volume_pct):
                breakout_occurred = True
                breakout_direction = 'short'

        return breakout_occurred, breakout_direction, breakout_distance, breakout_pct_of_or, current_idx

    else:
        if current_high > opening_high:  # Long breakout (touch-based)
            breakout_occurred = True
            breakout_direction = 'long'
        elif current_low < opening_low:  # Short breakout (touch-based)
            breakout_occurred = True
            breakout_direction = 'short'

        return breakout_occurred, breakout_direction, breakout_distance, breakout_pct_of_or, current_idx

def check_reversal(
    breakout_time: pd.Timestamp,
    current_idx: pd.Timestamp,
    next_close: float,
    next_volume: float,
    next_volume_ema21: float,
    breakout_direction: str,
    opening_high: float,
    opening_low: float,
    min_volume_pct: float
) -> Tuple[bool, str, int]:
    """
    Returns:
        reversal_valid (bool),
        entry_direction (long/short),
        bars_after_breakout (int)
    """
    # Calculate number of bars since breakout
    bars_after_breakout = int((current_idx - breakout_time).total_seconds() / 300)  # 5-minute bars

    volume_pct = next_volume / next_volume_ema21 if next_volume_ema21 > 0 else 0

    if breakout_direction == 'long':
        reversal_condition = next_close < opening_high
        entry_direction = 'short'
    else:
        reversal_condition = next_close > opening_low
        entry_direction = 'long'

    return reversal_condition and (volume_pct >= min_volume_pct), entry_direction, bars_after_breakout

def execute_breakout_trade(
    data: pd.DataFrame,
    breakout_row: pd.Series,
    breakout_idx: pd.Timestamp,
    entry_direction: str,
    opening_high: float,
    opening_low: float,
    buffer: float,
    cost: float,
    trade_template: Dict[str, Any],
    breakout_distance: float,
    breakout_pct_of_or: float,
    volume_pct: float,
    tp_value: float = 6.0,
    sl_value: float = 6.0
) -> Optional[Dict[str, Any]]:
    """
    Executes breakout trade on first valid breakout only.
    Enters at breakout candle close + buffer + cost.
    Fixed TP/SL of 6 points each.
    """
    breakout_close = safe_float(breakout_row['Close'])

    # Calculate entry price with buffer and cost
    if entry_direction == 'long':
        if breakout_requires_close_outside:
            entry_price = breakout_close + buffer + cost
        else:
            entry_price = opening_high + buffer + cost
        tp_price = entry_price + tp_value
        sl_price = entry_price - sl_value
    else:  # short
        if breakout_requires_close_outside:
            entry_price = breakout_close - buffer - cost
        else:
            entry_price = opening_low - buffer - cost
        tp_price = entry_price - tp_value
        sl_price = entry_price + sl_value

    # Create trade record
    trade = trade_template.copy()
    trade.update({
        'breakout_candle': breakout_idx,
        'breakout_distance': breakout_distance,
        'entry_time': breakout_idx,
        'entry_price': entry_price,
        'direction': entry_direction,
        'tp_price': tp_price,
        'sl_price': sl_price,
        'trade_taken': True,
        'entry_volume': safe_float(breakout_row['Volume']),
        'entry_volume_ema21': safe_float(data.loc[breakout_idx, 'Volume_EMA21']),
        'breakout_pct_of_or': breakout_pct_of_or * 100,
        'volume_pct_of_ema21': volume_pct * 100,
        'strategy_type': 'ORB'
    })

    # Monitor for exits in subsequent candles
    post_breakout_data = data[data.index > breakout_idx]

    for exit_idx, exit_row in post_breakout_data.iterrows():
        exit_high = safe_float(exit_row['High'])
        exit_low = safe_float(exit_row['Low'])

        if entry_direction == 'long':
            if exit_high >= tp_price:  # TP hit
                trade.update({
                    'exit_time': exit_idx,
                    'exit_price': tp_price,
                    'exit_reason': 'TP',
                    'pnl': tp_price - entry_price
                })
                break
            elif exit_low <= sl_price:  # SL hit
                trade.update({
                    'exit_time': exit_idx,
                    'exit_price': sl_price,
                    'exit_reason': 'SL',
                    'pnl': sl_price - entry_price
                })
                break
        else:  # short
            if exit_low <= tp_price:  # TP hit
                trade.update({
                    'exit_time': exit_idx,
                    'exit_price': tp_price,
                    'exit_reason': 'TP',
                    'pnl': entry_price - tp_price
                })
                break
            elif exit_high >= sl_price:  # SL hit
                trade.update({
                    'exit_time': exit_idx,
                    'exit_price': sl_price,
                    'exit_reason': 'SL',
                    'pnl': entry_price - sl_price
                })
                break
    else:
        # EOD exit if neither TP nor SL hit
        last_close = safe_float(post_breakout_data['Close'].iloc[-1])
        trade.update({
            'exit_time': post_breakout_data.index[-1],
            'exit_price': last_close,
            'exit_reason': 'EOD',
            'pnl': (last_close - entry_price) if entry_direction == 'long' else (entry_price - last_close)
        })

    trade['position_duration'] = trade['exit_time'] - trade['entry_time']
    return trade


def simulate_trade(
    data: pd.DataFrame,
    opening_high: float,
    opening_low: float,
    atr: float,
    expected_candles: int,
    or_start_hour: int,
    or_start_minute: str,
    or_end_hour: int,
    or_end_minute: str,
    buffer: float,
    cost: float,
    min_breakout_pct: float = 0.0,
    max_breakout_pct: float = 1.0,
    min_volume_pct: float = 0.9
) -> Optional[Dict[str, Any]]:
    """Takes only the first valid breakout trade per day"""
    data = data.copy()
    data['Volume_EMA21'] = data['Volume'].ewm(span=21, adjust=False).mean()

    or_start_min = int(or_start_minute)
    or_end_min = int(or_end_minute)
    opening_range_end_time = time(or_end_hour, or_end_min)
    post_opening_data = data[data.index.time > opening_range_end_time]

    # Initialize trade template
    trade_template = {
        'date': data.index[0].date(),
        'opening_high': opening_high,
        'opening_low': opening_low,
        'or_range': opening_high - opening_low,
        'direction': None,
        'entry_time': None,
        'entry_price': None,
        'exit_time': None,
        'exit_price': None,
        'exit_reason': None,
        'pnl': None,
        'trade_taken': False,
        'position_duration': None,
        'atr': round(atr, 2),
        'strategy_type': 'ORB'
    }

    # Track if we've already taken a trade today
    trade_taken_today = False

    for current_idx, current_row in post_opening_data.iterrows():
        if trade_taken_today:
            break  # Only take first trade per day

        breakout_occurred, breakout_dir, breakout_dist, breakout_pct, _ = check_breakout(
            current_idx=current_idx,
            current_low=safe_float(current_row['Low']),
            current_high=safe_float(current_row['High']),
            current_open=safe_float(current_row['Open']),
            current_close=safe_float(current_row['Close']),
            current_volume=safe_float(current_row['Volume']),
            current_volume_ema21=safe_float(current_row['Volume_EMA21']),
            opening_high=opening_high,
            opening_low=opening_low,
            min_breakout_pct=min_breakout_pct,
            max_breakout_pct=max_breakout_pct,
            min_volume_pct=min_volume_pct,
            opening_range_end_time=opening_range_end_time
        )

        if breakout_occurred:
            # Execute trade immediately at breakout candle close
            trade = execute_breakout_trade(
                data=data,
                breakout_row=current_row,
                breakout_idx=current_idx,
                entry_direction=breakout_dir,
                opening_high=opening_high,
                opening_low=opening_low,
                buffer=buffer,
                cost=cost,
                trade_template=trade_template,
                breakout_distance=breakout_dist,
                breakout_pct_of_or=breakout_pct,
                volume_pct=safe_float(current_row['Volume']) /
                          safe_float(current_row['Volume_EMA21']),
                tp_value=6.0,
                sl_value=6.0
            )
            trade_taken_today = True
            return trade

    return None

def execute_RB_trade(
    data: pd.DataFrame,
    current_row: pd.Series,
    next_row: pd.Series,
    next_idx: pd.Timestamp,
    entry_direction: str,
    opening_high: float,
    opening_low: float,
    or_midpoint: float,
    buffer: float,
    cost: float,
    trade_template: Dict[str, Any],
    breakout_candle: pd.Timestamp,
    breakout_distance: float,
    breakout_pct_of_or: float,
    volume_pct: float,
    bars_to_reversal: int
) -> Optional[Dict[str, Any]]:
    """
    Executes trade ONLY if entry condition (midpoint filter) is satisfied.
    Returns trade dict if valid, None otherwise.
    """
    next_close = safe_float(next_row['Close'])

    # Calculate entry price and check midpoint condition
    if entry_direction == 'long':
        entry_price = next_close + buffer + cost
        if entry_price >= or_midpoint:  # Fail midpoint condition
            return None
        tp_price = or_midpoint
        sl_price = safe_float(current_row['Low'])
    else:  # short
        entry_price = next_close - buffer - cost
        if entry_price <= or_midpoint:  # Fail midpoint condition
            return None
        tp_price = or_midpoint
        sl_price = safe_float(current_row['High'])

    # Proceed with trade execution
    trade = trade_template.copy()
    trade.update({
        'breakout_candle': breakout_candle,
        'breakout_distance': breakout_distance,
        'reversal_candle': next_idx,
        'entry_time': next_idx,
        'entry_price': entry_price,
        'direction': entry_direction,
        'tp_price': tp_price,
        'sl_price': sl_price,
        'trade_taken': True,
        'entry_volume': safe_float(next_row['Volume']),
        'entry_volume_ema21': safe_float(data.loc[next_idx, 'Volume_EMA21']),
        'next_volume_pct': (safe_float(next_row['Volume']) /
                          safe_float(data.loc[next_idx, 'Volume_EMA21'])) * 100,
        'bars_to_reversal': bars_to_reversal,
        'breakout_pct_of_or': breakout_pct_of_or * 100,
        'volume_pct_of_ema21': volume_pct * 100,
        'strategy_type': 'RB'
    })

    # Monitor for exits
    post_opening_data = data[data.index > breakout_candle]
    candle_times = list(post_opening_data.index)
    start_idx = list(candle_times).index(next_idx)

    for exit_idx in candle_times[start_idx + 1:]:
        exit_row = post_opening_data.loc[exit_idx]
        exit_high = safe_float(exit_row['High'])
        exit_low = safe_float(exit_row['Low'])

        if entry_direction == 'long':
            if exit_low <= sl_price:  # SL hit
                trade.update({
                    'exit_time': exit_idx,
                    'exit_price': sl_price,
                    'exit_reason': 'SL',
                    'pnl': sl_price - entry_price
                })
                break
            elif exit_high >= tp_price:  # TP hit
                trade.update({
                    'exit_time': exit_idx,
                    'exit_price': tp_price,
                    'exit_reason': 'TP',
                    'pnl': tp_price - entry_price
                })
                break
        else:  # short
            if exit_high >= sl_price:  # SL hit
                trade.update({
                    'exit_time': exit_idx,
                    'exit_price': sl_price,
                    'exit_reason': 'SL',
                    'pnl': entry_price - sl_price
                })
                break
            elif exit_low <= tp_price:  # TP hit
                trade.update({
                    'exit_time': exit_idx,
                    'exit_price': tp_price,
                    'exit_reason': 'TP',
                    'pnl': entry_price - tp_price
                })
                break
    else:
        # EOD exit
        last_close = safe_float(post_opening_data['Close'].iloc[-1])
        trade.update({
            'exit_time': post_opening_data.index[-1],
            'exit_price': last_close,
            'exit_reason': 'EOD',
            'pnl': (last_close - entry_price) if entry_direction == 'long' else (entry_price - last_close)
        })

    trade['position_duration'] = trade['exit_time'] - trade['entry_time']
    return trade


def simulate_rb_trade(
    data: pd.DataFrame,
    opening_high: float,
    opening_low: float,
    atr: float,
    expected_candles: int,
    or_start_hour: int,
    or_start_minute: str,
    or_end_hour: int,
    or_end_minute: str,
    buffer: float,
    cost: float,
    max_reversal_bars: int = 1,
    min_breakout_pct: float = 0.2,
    max_breakout_pct: float = 1.0,
    min_volume_pct: float = 0.9
) -> Optional[Dict[str, Any]]:
    """Main function with first-breakout-only logic"""
    data = data.copy()
    data['Volume_EMA21'] = data['Volume'].ewm(span=21, adjust=False).mean()

    or_start_min = int(or_start_minute)
    or_end_min = int(or_end_minute)
    opening_range_end_time = time(or_end_hour, or_end_min)
    post_opening_data = data[data.index.time > opening_range_end_time]
    or_midpoint = (opening_high + opening_low) / 2

    # Initialize trade template
    trade_template = {
        'date': data.index[0].date(),
        'opening_high': opening_high,
        'opening_low': opening_low,
        'or_range': opening_high - opening_low,
        'or_midpoint': or_midpoint,
        'direction': None,
        'entry_time': None,
        'entry_price': None,
        'exit_time': None,
        'exit_price': None,
        'exit_reason': None,
        'pnl': None,
        'trade_taken': False,
        'position_duration': None,
        'atr': round(atr, 2),
        'entry_volume': None,
        'entry_volume_ema21': None,
        'next_volume_pct': None,
        'breakout_distance': None,
        'breakout_candle': None,
        'reversal_candle': None,
        'bars_to_reversal': None,
        'strategy_type': 'RB',
        'breakout_pct_of_or': None,
        'volume_pct_of_ema21': None,
        'min_breakout_pct': min_breakout_pct * 100,
        'max_breakout_pct': max_breakout_pct * 100,
        'min_volume_pct': min_volume_pct * 100
    }

    candle_times = list(post_opening_data.index)
    breakout_info = None
    last_breakout_time = None

    for i in range(len(candle_times)):
        current_idx = candle_times[i]
        current_row = post_opening_data.loc[current_idx]

        # Check for new breakout if we don't have an active one
        if breakout_info is None:
            breakout_occurred, breakout_dir, breakout_dist, breakout_pct, breakout_time = check_breakout(
                current_idx=current_idx,
                current_low=safe_float(current_row['Low']),
                current_high=safe_float(current_row['High']),
                current_open=safe_float(current_row['Open']),
                current_close=safe_float(current_row['Close']),
                current_volume=safe_float(current_row['Volume']),
                current_volume_ema21=safe_float(current_row['Volume_EMA21']),
                opening_high=opening_high,
                opening_low=opening_low,
                min_breakout_pct=min_breakout_pct,
                max_breakout_pct=max_breakout_pct,
                min_volume_pct=min_volume_pct,
                opening_range_end_time=opening_range_end_time,
                last_breakout_time=last_breakout_time
            )

            if breakout_occurred:
                breakout_info = (breakout_dir, breakout_dist, breakout_pct, breakout_time)
                last_breakout_time = breakout_time
                continue  # Move to next candle to check for reversal

        # If we have an active breakout, check for reversal
        if breakout_info is not None:
            breakout_dir, breakout_dist, breakout_pct, breakout_time = breakout_info

            # Check if we're still within max_reversal_bars
            bars_since_breakout = int((current_idx - breakout_time).total_seconds() / 300)
            if bars_since_breakout > max_reversal_bars:
                breakout_info = None  # Reset if too many bars passed
                continue

            reversal_valid, entry_dir, bars_after_breakout = check_reversal(
                breakout_time=breakout_time,
                current_idx=current_idx,
                next_close=safe_float(current_row['Close']),
                next_volume=safe_float(current_row['Volume']),
                next_volume_ema21=safe_float(data.loc[current_idx, 'Volume_EMA21']),
                breakout_direction=breakout_dir,
                opening_high=opening_high,
                opening_low=opening_low,
                min_volume_pct=min_volume_pct
            )

            if reversal_valid:
                # Execute trade if conditions met
                trade = execute_RB_trade(
                    data=data,
                    current_row=post_opening_data.loc[breakout_time],  # Breakout candle
                    next_row=current_row,  # Reversal candle
                    next_idx=current_idx,
                    entry_direction=entry_dir,
                    opening_high=opening_high,
                    opening_low=opening_low,
                    or_midpoint=or_midpoint,
                    buffer=buffer,
                    cost=cost,
                    trade_template=trade_template,
                    breakout_candle=breakout_time,
                    breakout_distance=breakout_dist,
                    breakout_pct_of_or=breakout_pct,
                    volume_pct=safe_float(post_opening_data.loc[breakout_time, 'Volume']) /
                              safe_float(post_opening_data.loc[breakout_time, 'Volume_EMA21']),
                    bars_to_reversal=bars_after_breakout
                )

                if trade:  # Only return if execute_RB_trade approved the entry
                    breakout_info = None  # Reset breakout after trade
                    return trade

    return None

def calculate_vwap(data):
    """Calculate Volume-Weighted Average Price (VWAP) using HLC3 as price input"""
    hlc3 = (data['High'] + data['Low'] + data['Close']) / 3  # HLC3 instead of typical Close
    vwap = (hlc3 * data['Volume']).cumsum() / data['Volume'].cumsum()
    return vwap

def plot_single_day(data: pd.DataFrame, date: datetime.date,
                   opening_high: Optional[float], opening_low: Optional[float],
                   trade: Optional[Dict] = None, timezone='Europe/London'):
    """Plot single trading day with VWAP, EMAs, volume, and ATR"""
    # Calculate all indicators
    hlc3 = (data['High'] + data['Low'] + data['Close']) / 3
    data['VWAP'] = (hlc3 * data['Volume']).cumsum() / data['Volume'].cumsum()
    data['EMA21'] = data['Close'].ewm(span=21, adjust=False).mean()
    data['EMA200'] = data['Close'].ewm(span=200, adjust=False).mean()
    data['Volume_EMA21'] = data['Volume'].ewm(span=21, adjust=False).mean()

    # Prepare OHLC data with color coding
    ohlc = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    ohlc['Color'] = np.where(ohlc['Close'] >= ohlc['Open'], 'green', 'red')
    ohlc = ohlc.reset_index()
    ohlc['Date_mpl'] = ohlc['Datetime'].apply(lambda x: mdates.date2num(x.to_pydatetime()))

    # Create figure with 3 subplots
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[4, 1, 1])
    ax1 = fig.add_subplot(gs[0])  # Price with EMAs
    ax2 = fig.add_subplot(gs[1])  # Volume
    ax3 = fig.add_subplot(gs[2])  # ATR

    # --- Price Chart ---
    # Candlesticks
    candlestick_ohlc(
        ax=ax1,
        quotes=ohlc[['Date_mpl', 'Open', 'High', 'Low', 'Close']].values,
        width=0.0015,
        colorup='g',
        colordown='r',
        alpha=1
    )

    # Moving Averages
    ax1.plot(ohlc['Date_mpl'], data['EMA21'], color='blue', linewidth=1.5, label='EMA 21')
    ax1.plot(ohlc['Date_mpl'], data['EMA200'], color='red', linewidth=1.5, label='EMA 200')

    # VWAP line (yellow)
    ax1.plot(ohlc['Date_mpl'], data['VWAP'], color='yellow', linewidth=2, label='VWAP (HLC3)')

    # Opening range levels
    if opening_high is not None:
        ax1.axhline(y=opening_high, color='blue', linestyle='--', label=f'OR High: {opening_high:.2f}')
    if opening_low is not None:
        ax1.axhline(y=opening_low, color='purple', linestyle='--', label=f'OR Low: {opening_low:.2f}')
        or_midpoint=(opening_high + opening_low)/2
        ax1.axhline(y=or_midpoint, color='gray', linestyle=':', alpha=0.7, label=f'OR Midpoint: {or_midpoint:.2f}')

    # Trade annotations
    if trade is not None:
        entry_time = mdates.date2num(trade['entry_time'].to_pydatetime())
        exit_time = mdates.date2num(trade['exit_time'].to_pydatetime())

        ax1.plot(entry_time, trade['entry_price'], 'bo', markersize=10, label='Entry')
        ax1.plot(exit_time, trade['exit_price'], 'ro', markersize=10, label='Exit')
        ax1.plot([entry_time, exit_time],
                [trade['entry_price'], trade['exit_price']],
                'k-', linewidth=2)

        trade_summary = (
            f"Trade Executed:\n"
            f"Direction: {trade['direction']}\n"
            f"Entry Time: {trade['entry_time'].strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
            f"Entry Price: ${trade['entry_price']:.2f}\n"
            f"Exit Time: {trade['exit_time'].strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
            f"Exit Price: ${trade['exit_price']:.2f}\n"
            f"Exit Reason: {trade['exit_reason']}\n"
            f"P/L: {trade['pnl']:.2f} points"
        )

        ax1.annotate(trade_summary,
                    xy=(exit_time, trade['exit_price']),
                    xytext=(50, 30),
                    textcoords='offset points',
                    fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.5', fc='lightyellow', alpha=0.9),
                    arrowprops=dict(arrowstyle='->', color='black'))

    # --- Volume Chart ---
    # Colored volume bars
    for idx, row in ohlc.iterrows():
        ax2.bar(row['Date_mpl'], row['Volume'], width=0.001, color=row['Color'], alpha=0.7)

    # Volume EMA 21
    ax2.plot(ohlc['Date_mpl'], data['Volume_EMA21'], color='blue', linewidth=1.5, label='Volume EMA 21')
    ax2.set_ylabel('Volume')
    ax2.legend(loc='upper left')

    # --- ATR Chart ---
    if trade is not None:
        ax3.axhline(y=trade['atr'], color='green', linestyle='-', label='ATR')
        ax3.axhline(y=trade['tp_distance'], color='blue', linestyle='--', label='TP Distance')
        ax3.axhline(y=trade['sl_distance'], color='red', linestyle='--', label='SL Distance')
        ax3.set_ylabel('Points')
        ax3.legend()
        ax3.set_title('ATR and Stop Levels')

    # Formatting
    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=pytz.timezone(timezone)))
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=15))
        ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=5))

    start_dt = pytz.timezone(timezone).localize(
        datetime.combine(date, datetime.strptime(f'{start_hour}:00', '%H:%M').time()))
    end_dt = pytz.timezone(timezone).localize(
        datetime.combine(date, datetime.strptime(f'{end_hour}:{end_minute}', '%H:%M').time()))

    margin = timedelta(minutes=5)
    x_min = mdates.date2num(start_dt - margin)
    x_max = mdates.date2num(end_dt + margin)

    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(x_min, x_max)

    # Hide x-axis labels for upper plots
    # ax1.set_xticklabels([])
    # ax2.set_xticklabels([])

    ax1.grid(True, which='major', linestyle='-', alpha=0.7)
    ax1.set_title(f'{ticker} 5-minute Chart ({date})')
    ax1.legend(loc='upper left', fontsize='small')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

def plot_optimization_heatmap(results_df: pd.DataFrame):
    """Create a heatmap of absolute TP/SL optimization results"""
    heatmap_data = results_df.pivot(index='sl_value', columns='tp_value', values='total_pnl')

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".1f",
        cmap='RdYlGn',
        center=0,
        cbar_kws={'label': 'Total P/L (points)'},
        ax=ax
    )
    ax.set_title('TP/SL Optimization Results (Absolute Values)')
    ax.set_xlabel('Take Profit (points)')
    ax.set_ylabel('Stop Loss (points)')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

def plot_win_rate_heatmap(results_df: pd.DataFrame):
    """Create a heatmap of win rate optimization results"""
    heatmap_data = results_df.pivot(index='sl_value', columns='tp_value', values='win_rate')

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".1f",
        cmap='RdYlGn',
        vmin=0,  # Win rate can't be negative
        vmax=100,  # Maximum win rate is 100%
        cbar_kws={'label': 'Win Rate (%)'},
        ax=ax
    )
    ax.set_title('Win Rate Optimization Results')
    ax.set_xlabel('Take Profit (points)')
    ax.set_ylabel('Stop Loss (points)')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

def plot_efficiency_frontier(results_df: pd.DataFrame):
    """Create an efficiency frontier plot showing win rate vs total P/L"""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create color mapping based on risk-reward ratio (TP/SL)
    results_df['risk_reward'] = results_df['tp_value'] / results_df['sl_value']
    colors = results_df['risk_reward']

    # Plot all points with color indicating risk-reward ratio
    scatter = ax.scatter(
        x=results_df['win_rate'],
        y=results_df['total_pnl'],
        c=colors,
        cmap='viridis',
        s=100,
        alpha=0.7,
        edgecolors='w',
        linewidths=0.5
    )

    # Highlight Pareto efficient points (frontier)
    sorted_df = results_df.sort_values('win_rate')
    pareto_front = []
    current_max = -np.inf
    for _, row in sorted_df.iterrows():
        if row['total_pnl'] > current_max:
            pareto_front.append(row)
            current_max = row['total_pnl']
    pareto_df = pd.DataFrame(pareto_front)

    ax.plot(
        pareto_df['win_rate'],
        pareto_df['total_pnl'],
        'r--',
        alpha=0.7,
        label='Efficiency Frontier'
    )

    # Add labels and annotations for interesting points
    max_pnl_idx = results_df['total_pnl'].idxmax()
    max_pnl_point = results_df.loc[max_pnl_idx]
    ax.scatter(
        x=max_pnl_point['win_rate'],
        y=max_pnl_point['total_pnl'],
        c='red',
        s=200,
        marker='s',
        label=f"Max P/L (TP={max_pnl_point['tp_value']}, SL={max_pnl_point['sl_value']})"
    )

    max_win_rate_idx = results_df['win_rate'].idxmax()
    max_win_rate_point = results_df.loc[max_win_rate_idx]
    ax.scatter(
        x=max_win_rate_point['win_rate'],
        y=max_win_rate_point['total_pnl'],
        c='blue',
        s=200,
        marker='o',
        label=f"Max Win Rate (TP={max_win_rate_point['tp_value']}, SL={max_win_rate_point['sl_value']})"
    )

    # Add colorbar for risk-reward ratio
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Risk-Reward Ratio (TP/SL)')

    # Formatting
    ax.set_xlabel('Win Rate (%)')
    ax.set_ylabel('Total P/L (points)')
    ax.set_title('Efficiency Frontier: Win Rate vs Total P/L')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

def analyze_strategy_performance(trades_df: pd.DataFrame) -> Dict:
    """Calculate strategy performance metrics from the trades DataFrame"""
    if trades_df.empty:
        return {}

    # Overall metrics
    metrics = {
        'total_trades': len(trades_df),
        'winning_trades': len(trades_df[trades_df['pnl'] > 0]),
        'losing_trades': len(trades_df[trades_df['pnl'] < 0]),
        'win_rate': len(trades_df[trades_df['pnl'] > 0]) / len(trades_df) * 100,
        'total_pnl': trades_df['pnl'].sum(),
        'avg_pnl': trades_df['pnl'].mean(),
        'avg_trade_duration': trades_df['position_duration'].mean(),
        'max_win': trades_df['pnl'].max(),
        'max_loss': trades_df['pnl'].min(),
        'profit_factor': abs(trades_df[trades_df['pnl'] > 0]['pnl'].sum() /
                           trades_df[trades_df['pnl'] < 0]['pnl'].sum())
    }

    # Long positions metrics
    long_trades = trades_df[trades_df['direction'] == 'long']
    if not long_trades.empty:
        metrics.update({
            'long_total_trades': len(long_trades),
            'long_winning_trades': len(long_trades[long_trades['pnl'] > 0]),
            'long_losing_trades': len(long_trades[long_trades['pnl'] < 0]),
            'long_win_rate': len(long_trades[long_trades['pnl'] > 0]) / len(long_trades) * 100,
            'long_total_pnl': long_trades['pnl'].sum(),
            'long_avg_pnl': long_trades['pnl'].mean(),
            'long_avg_trade_duration': long_trades['position_duration'].mean(),
            'long_max_win': long_trades['pnl'].max(),
            'long_max_loss': long_trades['pnl'].min(),
            'long_profit_factor': abs(long_trades[long_trades['pnl'] > 0]['pnl'].sum() /
                                    long_trades[long_trades['pnl'] < 0]['pnl'].sum())
        })

    # Short positions metrics
    short_trades = trades_df[trades_df['direction'] == 'short']
    if not short_trades.empty:
        metrics.update({
            'short_total_trades': len(short_trades),
            'short_winning_trades': len(short_trades[short_trades['pnl'] > 0]),
            'short_losing_trades': len(short_trades[short_trades['pnl'] < 0]),
            'short_win_rate': len(short_trades[short_trades['pnl'] > 0]) / len(short_trades) * 100,
            'short_total_pnl': short_trades['pnl'].sum(),
            'short_avg_pnl': short_trades['pnl'].mean(),
            'short_avg_trade_duration': short_trades['position_duration'].mean(),
            'short_max_win': short_trades['pnl'].max(),
            'short_max_loss': short_trades['pnl'].min(),
            'short_profit_factor': abs(short_trades[short_trades['pnl'] > 0]['pnl'].sum() /
                                  short_trades[short_trades['pnl'] < 0]['pnl'].sum())
        })

    return metrics

def calculate_percentage_volume(row):
    """Helper to calculate volume as percentage of its EMA21"""
    if row['entry_volume_ema21'] > 0:
        return (row['entry_volume'] / row['entry_volume_ema21']) * 100
    return 0

def calculate_breakout_pct_of_or(row):
    """Helper to calculate breakout distance as percentage of OR range"""
    if row['or_range'] > 0:
        return (row['breakout_distance'] / row['or_range']) * 100
    return 0

def format_duration_minutes(td):
    """Helper function to format timedelta as minutes"""
    if pd.isna(td):
        return "N/A"
    return f"{td.total_seconds()/60:.1f}"

# Main execution block
# Main execution block - Download/Upload Section

def prepare_download_data(data: pd.DataFrame) -> Tuple[bytes, str]:
    """Prepares data for download with European timezone timestamps"""
    # Make a copy to avoid modifying original
    download_data = data.copy()

    # Convert to Europe timezone but keep as naive datetime (remove tzinfo)
    if download_data.index.tz is not None:
        download_data.index = download_data.index.tz_convert(timezone).tz_localize(None)

    # Reorder columns to match your preferred format
    download_data = download_data[['Open', 'High', 'Low', 'Close', 'Volume']]

    # Format dates in European style (DD/MM/YYYY)
    download_data.index = download_data.index.strftime('%d/%m/%Y %H:%M')

    # Generate filename with parameters
    safe_timezone = timezone.replace('/', '_')
    file_name = f"{ticker}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}_{safe_timezone}.csv"

    # Convert to CSV bytes
    csv_data = download_data.to_csv().encode('utf-8')

    return csv_data, file_name

def process_uploaded_data(uploaded_file, timezone: str, start_date, end_date) -> pd.DataFrame:
    """Processes uploaded CSV with European timestamps"""
    try:
        # Read CSV skipping the first two rows (header starts at row 3)
        full_data = pd.read_csv(uploaded_file, index_col=0, parse_dates=True, dayfirst=True, skiprows=2)

        # Rename columns to match expected format
        full_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

        # Validate required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in full_data.columns for col in required_cols):
            missing = [col for col in required_cols if col not in full_data.columns]
            raise ValueError(f"Missing required columns: {missing}")

        # Convert naive datetime to localized (assume CSV times are in Europe timezone)
        full_data.index = pd.to_datetime(full_data.index).tz_localize(timezone)

        # Filter date range
        start_dt = pd.to_datetime(start_date).tz_localize(timezone)
        end_dt = pd.to_datetime(end_date).tz_localize(timezone) + timedelta(days=1)
        full_data = full_data[(full_data.index >= start_dt) & (full_data.index <= end_dt)]

        if full_data.empty:
            raise ValueError("No data in selected date range")

        return full_data

    except Exception as e:
        st.error(f"CSV processing error: {str(e)}")
        st.stop()

# Download button
if st.sidebar.button("Download Processed Data"):
    with st.spinner("Downloading and processing data..."):
        try:
            processed_data = get_trading_days_data(
                start_date=start_date,
                end_date=end_date,
                ticker=ticker,
                timezone=timezone
            )

            if not processed_data.empty:
                st.success(f"Downloaded {len(processed_data)} bars of data")

                # Show preview in European date format
                preview_data = processed_data.copy()
                preview_data.index = preview_data.index.strftime('%d/%m/%Y %H:%M')
                st.dataframe(preview_data[['Open', 'High', 'Low', 'Close', 'Volume']].head())

                csv_data, file_name = prepare_download_data(processed_data)

                st.download_button(
                    label="Download Processed Data as CSV",
                    data=csv_data,
                    file_name=file_name,
                    mime='text/csv',
                    help="Contains timezone-converted data in UTC format"
                )
            else:
                st.error("No data available for download")

        except Exception as e:
            st.error(f"Download failed: {str(e)}")

# Run Analysis button
if st.sidebar.button("Run Analysis"):
    with st.spinner("Running analysis..."):
        if data_source == "YFinance":
            full_data = get_trading_days_data(start_date, end_date, ticker, timezone)
            if full_data.empty:
                st.error("No Yahoo Finance data available!")
                st.stop()
        else:  # CSV upload
            if uploaded_file is None:
                st.error("Please upload a CSV file first!")
                st.stop()

            full_data = process_uploaded_data(
                uploaded_file=uploaded_file,
                timezone=timezone,
                start_date=start_date,
                end_date=end_date
            )

        # Read and process the uploaded CSV
        daily_data = split_data_by_day(full_data, start_hour, start_minute, end_hour, end_minute)
        st.success(f"Found {len(daily_data)} trading days in the date range")

        # Generate trades
        all_trades = []
        for date, day_data in daily_data.items():
            high, low, tr, expected_candles = get_opening_range_levels(
                day_data, or_start_hour, or_start_minute, or_end_hour, or_end_minute,
                start_hour, start_minute, end_hour, end_minute
            )
            if None in (high, low, tr):
                continue

            trades_for_day = []

            # Run ORB if selected
            if strategy_type in ["ORB", "Both"]:
                orb_trade = simulate_trade(
                    data=day_data,
                    opening_high=high,
                    opening_low=low,
                    atr=tr,
                    expected_candles=expected_candles,
                    or_start_hour=or_start_hour,
                    or_start_minute=or_start_minute,
                    or_end_hour=or_end_hour,
                    or_end_minute=or_end_minute,
                    buffer=buffer_pts,
                    cost=cost_pts,
                    min_breakout_pct=min_breakout_pct,
                    max_breakout_pct=max_breakout_pct,
                    min_volume_pct=min_volume_pct
                )
                if orb_trade:
                    trades_for_day.append(orb_trade)

            # Run RB if selected
            if strategy_type in ["RB", "Both"]:
                rb_trade = simulate_rb_trade(
                    data=day_data,
                    opening_high=high,
                    opening_low=low,
                    atr=tr,
                    expected_candles=expected_candles,
                    or_start_hour=or_start_hour,
                    or_start_minute=or_start_minute,
                    or_end_hour=or_end_hour,
                    or_end_minute=or_end_minute,
                    buffer=buffer_pts,
                    cost=cost_pts,
                    max_reversal_bars=max_reversal_bars,
                    min_breakout_pct=min_breakout_pct,
                    max_breakout_pct=max_breakout_pct,
                    min_volume_pct=min_volume_pct
                )
                if rb_trade:
                    trades_for_day.append(rb_trade)

            all_trades.extend(trades_for_day)

        # Create trades DataFrame if trades exist
        if all_trades:
            trades_df = pd.DataFrame(all_trades)

            # Convert and format columns
            if 'date' in trades_df.columns and trades_df['date'].dtype == object:
                trades_df['date'] = pd.to_datetime(trades_df['date']).dt.date

            if 'position_duration' in trades_df.columns:
                trades_df['position_duration'] = pd.to_timedelta(trades_df['position_duration'])

            # Calculate TP/SL distances for display
            trades_df['tp_distance'] = abs(trades_df['tp_price'] - trades_df['entry_price'])
            trades_df['sl_distance'] = abs(trades_df['sl_price'] - trades_df['entry_price'])

            # Calculate risk-reward ratio
            trades_df['rr_ratio'] = trades_df['tp_distance'] / trades_df['sl_distance']

            # Display trade details with updated columns
            st.subheader("Trade Details")

            # Updated display columns for rubber band strategy
            display_columns = [
                'date', 'direction', 'entry_time', 'exit_time',
                'entry_price', 'exit_price', 'pnl', 'exit_reason',
                'position_duration', 'atr',
                'tp_price', 'sl_price',  # Actual price levels
                'tp_distance', 'sl_distance', 'rr_ratio',  # Calculated values
                'opening_high', 'opening_low', 'or_range',
                'entry_volume', 'entry_volume_ema21',
                'breakout_distance', 'breakout_candle', 'reversal_candle',
                'strategy_type'
            ]

            # Filter only columns that exist in the DataFrame
            existing_columns = [col for col in display_columns if col in trades_df.columns]
            formatted_trades = trades_df[existing_columns].copy()

            # Format specific columns
            formatted_trades['position_duration_min'] = formatted_trades['position_duration'].apply(
                lambda x: f"{x.total_seconds()/60:.1f} min" if pd.notnull(x) else "N/A"
            )

            # Rename columns for better display
            formatted_trades = formatted_trades.rename(columns={
                'date': 'Date',
                'direction': 'Direction',
                'entry_time': 'Entry Time',
                'exit_time': 'Exit Time',
                'entry_price': 'Entry Price',
                'exit_price': 'Exit Price',
                'pnl': 'P/L (pts)',
                'exit_reason': 'Exit Reason',
                'position_duration_min': 'Duration',
                'atr': 'ATR',
                'tp_price': 'TP Price',
                'sl_price': 'SL Price',
                'tp_distance': 'TP Dist',
                'sl_distance': 'SL Dist',
                'rr_ratio': 'R/R Ratio',
                'opening_high': 'OR High',
                'opening_low': 'OR Low',
                'or_range': 'OR Range',
                'entry_volume': 'Entry Vol',
                'entry_volume_ema21': 'Vol EMA21',
                'breakout_distance': 'Breakout Dist',
                'breakout_candle': 'Breakout Time',
                'reversal_candle': 'Reversal Time',
                'strategy_type': 'Strategy'
            })

            # Reorder columns for logical display
            final_columns = [
                'Date', 'Direction', 'Entry Time', 'Exit Time', 'Duration',
                'Entry Price', 'Exit Price', 'P/L (pts)', 'Exit Reason',
                'TP Price', 'SL Price', 'TP Dist', 'SL Dist', 'R/R Ratio',
                'OR High', 'OR Low', 'OR Range', 'Breakout Dist',
                'Breakout Time', 'Reversal Time', 'Entry Vol', 'Vol EMA21', 'ATR'
            ]

            # Filter only columns that exist after renaming
            final_columns = [col for col in final_columns if col in formatted_trades.columns]

            st.dataframe(formatted_trades[final_columns])

            # Performance analysis
            metrics = analyze_strategy_performance(trades_df)

            st.subheader("Strategy Performance")

            # Overall Performance
            st.markdown("### Overall Performance")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Trades", metrics.get('total_trades', 0))
                st.metric("Winning Trades", metrics.get('winning_trades', 0))
                st.metric("Losing Trades", metrics.get('losing_trades', 0))
                st.metric("Win Rate", f"{metrics.get('win_rate', 0):.1f}%")
            with col2:
                st.metric("Total PNL", f"{metrics.get('total_pnl', 0):.2f}")
                st.metric("Average PNL", f"{metrics.get('avg_pnl', 0):.2f}")
                st.metric("Average Trade Duration", format_duration_minutes(metrics.get('avg_trade_duration')))
                st.metric("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}")

            # Long Positions Performance
            st.markdown("### Long Positions Performance")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Long Trades", metrics.get('long_total_trades', 0))
                st.metric("Winning Long Trades", metrics.get('long_winning_trades', 0))
                st.metric("Losing Long Trades", metrics.get('long_losing_trades', 0))
                st.metric("Long Win Rate", f"{metrics.get('long_win_rate', 0):.1f}%")
            with col2:
                st.metric("Long Total PNL", f"{metrics.get('long_total_pnl', 0):.2f}")
                st.metric("Long Average PNL", f"{metrics.get('long_avg_pnl', 0):.2f}")
                st.metric("Long Average Duration", format_duration_minutes(metrics.get('long_avg_trade_duration')))
                st.metric("Long Profit Factor", f"{metrics.get('long_profit_factor', 0):.2f}")

            # Short Positions Performance
            st.markdown("### Short Positions Performance")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Short Trades", metrics.get('short_total_trades', 0))
                st.metric("Winning Short Trades", metrics.get('short_winning_trades', 0))
                st.metric("Losing Short Trades", metrics.get('short_losing_trades', 0))
                st.metric("Short Win Rate", f"{metrics.get('short_win_rate', 0):.1f}%")
            with col2:
                st.metric("Short Total PNL", f"{metrics.get('short_total_pnl', 0):.2f}")
                st.metric("Short Average PNL", f"{metrics.get('short_avg_pnl', 0):.2f}")
                st.metric("Short Average Duration", format_duration_minutes(metrics.get('short_avg_trade_duration')))
                st.metric("Short Profit Factor", f"{metrics.get('short_profit_factor', 0):.2f}")


            # Add this right before the cumulative PNL curve section in your code

            # Add OR Entry Summary section
            st.subheader("OR Entry Summary")

            if not trades_df.empty:
                # Calculate metrics
                trades_df['volume_pct_of_ema21'] = trades_df.apply(calculate_percentage_volume, axis=1)
                trades_df['breakout_pct_of_or'] = trades_df.apply(calculate_breakout_pct_of_or, axis=1)

                # Split into winning and losing trades
                winning_trades = trades_df[trades_df['pnl'] > 0]
                losing_trades = trades_df[trades_df['pnl'] < 0]

                # Create summary statistics
                def create_summary_table(df, title):
                    if df.empty:
                        return pd.DataFrame()

                    # Calculate stats for all trades
                    all_stats = {
                        'range_size': [df['or_range'].min(), df['or_range'].mean(), df['or_range'].max()],
                        'volume_pct': [df['volume_pct_of_ema21'].min(), df['volume_pct_of_ema21'].mean(), df['volume_pct_of_ema21'].max()],
                        'breakout_dist': [df['breakout_distance'].min(), df['breakout_distance'].mean(), df['breakout_distance'].max()],
                        'breakout_pct': [df['breakout_pct_of_or'].min(), df['breakout_pct_of_or'].mean(), df['breakout_pct_of_or'].max()]
                    }

                    # Calculate stats for long trades
                    long_trades = df[df['direction'] == 'long']
                    long_stats = {
                        'range_size': [long_trades['or_range'].min(), long_trades['or_range'].mean(), long_trades['or_range'].max()] if not long_trades.empty else [0, 0, 0],
                        'volume_pct': [long_trades['volume_pct_of_ema21'].min(), long_trades['volume_pct_of_ema21'].mean(), long_trades['volume_pct_of_ema21'].max()] if not long_trades.empty else [0, 0, 0],
                        'breakout_dist': [long_trades['breakout_distance'].min(), long_trades['breakout_distance'].mean(), long_trades['breakout_distance'].max()] if not long_trades.empty else [0, 0, 0],
                        'breakout_pct': [long_trades['breakout_pct_of_or'].min(), long_trades['breakout_pct_of_or'].mean(), long_trades['breakout_pct_of_or'].max()] if not long_trades.empty else [0, 0, 0]
                    }

                    # Calculate stats for short trades
                    short_trades = df[df['direction'] == 'short']
                    short_stats = {
                        'range_size': [short_trades['or_range'].min(), short_trades['or_range'].mean(), short_trades['or_range'].max()] if not short_trades.empty else [0, 0, 0],
                        'volume_pct': [short_trades['volume_pct_of_ema21'].min(), short_trades['volume_pct_of_ema21'].mean(), short_trades['volume_pct_of_ema21'].max()] if not short_trades.empty else [0, 0, 0],
                        'breakout_dist': [short_trades['breakout_distance'].min(), short_trades['breakout_distance'].mean(), short_trades['breakout_distance'].max()] if not short_trades.empty else [0, 0, 0],
                        'breakout_pct': [short_trades['breakout_pct_of_or'].min(), short_trades['breakout_pct_of_or'].mean(), short_trades['breakout_pct_of_or'].max()] if not short_trades.empty else [0, 0, 0]
                    }

                    # Create DataFrame
                    summary = pd.DataFrame({
                        'Metric': ['Range Size', 'Volume % of EMA21', 'Breakout Distance', 'Breakout % of OR'],
                        'All Trades Min': [all_stats['range_size'][0], all_stats['volume_pct'][0], all_stats['breakout_dist'][0], all_stats['breakout_pct'][0]],
                        'All Trades Avg': [all_stats['range_size'][1], all_stats['volume_pct'][1], all_stats['breakout_dist'][1], all_stats['breakout_pct'][1]],
                        'All Trades Max': [all_stats['range_size'][2], all_stats['volume_pct'][2], all_stats['breakout_dist'][2], all_stats['breakout_pct'][2]],
                        'Long Trades Min': [long_stats['range_size'][0], long_stats['volume_pct'][0], long_stats['breakout_dist'][0], long_stats['breakout_pct'][0]],
                        'Long Trades Avg': [long_stats['range_size'][1], long_stats['volume_pct'][1], long_stats['breakout_dist'][1], long_stats['breakout_pct'][1]],
                        'Long Trades Max': [long_stats['range_size'][2], long_stats['volume_pct'][2], long_stats['breakout_dist'][2], long_stats['breakout_pct'][2]],
                        'Short Trades Min': [short_stats['range_size'][0], short_stats['volume_pct'][0], short_stats['breakout_dist'][0], short_stats['breakout_pct'][0]],
                        'Short Trades Avg': [short_stats['range_size'][1], short_stats['volume_pct'][1], short_stats['breakout_dist'][1], short_stats['breakout_pct'][1]],
                        'Short Trades Max': [short_stats['range_size'][2], short_stats['volume_pct'][2], short_stats['breakout_dist'][2], short_stats['breakout_pct'][2]]
                    })

                    summary.set_index('Metric', inplace=True)
                    return summary

                # Create tabs for Win/Loss breakdown
                tab1, tab2 = st.tabs(["Winning Trades", "Losing Trades"])

                with tab1:
                    st.markdown(f"**Winning Trades ({len(winning_trades)} trades)**")
                    if not winning_trades.empty:
                        win_summary = create_summary_table(winning_trades, "Winning Trades")
                        st.dataframe(win_summary.style.format({
                            'All Trades Min': '{:.2f}',
                            'All Trades Avg': '{:.2f}',
                            'All Trades Max': '{:.2f}',
                            'Long Trades Min': '{:.2f}',
                            'Long Trades Avg': '{:.2f}',
                            'Long Trades Max': '{:.2f}',
                            'Short Trades Min': '{:.2f}',
                            'Short Trades Avg': '{:.2f}',
                            'Short Trades Max': '{:.2f}'
                        }))
                    else:
                        st.write("No winning trades to display")

                with tab2:
                    st.markdown(f"**Losing Trades ({len(losing_trades)} trades)**")
                    if not losing_trades.empty:
                        loss_summary = create_summary_table(losing_trades, "Losing Trades")
                        st.dataframe(loss_summary.style.format({
                            'All Trades Min': '{:.2f}',
                            'All Trades Avg': '{:.2f}',
                            'All Trades Max': '{:.2f}',
                            'Long Trades Min': '{:.2f}',
                            'Long Trades Avg': '{:.2f}',
                            'Long Trades Max': '{:.2f}',
                            'Short Trades Min': '{:.2f}',
                            'Short Trades Avg': '{:.2f}',
                            'Short Trades Max': '{:.2f}'
                        }))
                    else:
                        st.write("No losing trades to display")
            else:
                st.write("No trades available for analysis")

            # Plot cumulative PNL
            st.subheader("Cumulative PNL Curve")
            fig, ax = plt.subplots(figsize=(12, 6))

            # Calculate cumulative PNL for all trades
            trades_df['cum_pnl'] = trades_df['pnl'].cumsum()
            ax.plot(pd.to_datetime(trades_df['date']), trades_df['cum_pnl'], 'b-', linewidth=2, label='All Trades')

            # Calculate cumulative PNL for long trades only
            long_trades = trades_df[trades_df['direction'] == 'long'].copy()
            if not long_trades.empty:
                long_trades['cum_pnl'] = long_trades['pnl'].cumsum()
                ax.plot(pd.to_datetime(long_trades['date']), long_trades['cum_pnl'], 'g-', linewidth=2, label='Long Trades Only')

            # Calculate cumulative PNL for short trades only
            short_trades = trades_df[trades_df['direction'] == 'short'].copy()
            if not short_trades.empty:
                short_trades['cum_pnl'] = short_trades['pnl'].cumsum()
                ax.plot(pd.to_datetime(short_trades['date']), short_trades['cum_pnl'], 'r-', linewidth=2, label='Short Trades Only')

            ax.set_title('Cumulative PNL Curve')
            ax.set_xlabel('Date')
            ax.set_ylabel('Cumulative PNL')
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)
            plt.close()

            # Display all trades in full size, newest first
            st.subheader("All Trade Executions")

            # Sort trades newest to oldest
            trades_df = trades_df.sort_values('date', ascending=False)

            # Create a container for all plots
            plots_container = st.container()

            with plots_container:
                for _, trade_row in trades_df.iterrows():
                    trade = trade_row.to_dict()
                    date = trade['date']
                    strategy_type = trade.get('strategy_type', 'N/A')  # Get strategy type or default to 'N/A'

                    # Get corresponding day data
                    if isinstance(date, pd.Timestamp):
                        date = date.date()
                    day_data = daily_data[date]

                    # Recompute opening range for plotting
                    high, low, tr, expected_candles = get_opening_range_levels(
                        day_data, or_start_hour, or_start_minute, or_end_hour, or_end_minute,
                        start_hour, start_minute, end_hour, end_minute
                    )

                    # Display date and P/L as a header
                    st.markdown(f"### {date.strftime('%Y-%m-%d')} - {strategy_type} - P/L: {trade['pnl']:.2f} points")

                    # Plot the trade with original full-size function
                    plot_single_day(day_data, date, high, low, trade, timezone)

                    # Add some spacing between plots
                    st.write("---")
