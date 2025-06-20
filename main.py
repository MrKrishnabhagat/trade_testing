#!/usr/bin/env python3
"""
NIFTY Options Intraday Trading System - REAL-TIME SIMULATION MODE
Uses live market data from KiteConnect API to test trading logic without placing actual orders

🔧 SIMULATION FEATURES:
1. ✅ REAL-TIME MARKET DATA: Live prices from KiteConnect API
2. ✅ ORDER RECORDING: All orders logged instead of placed
3. ✅ PROFIT-BASED TRAILING: Complete trailing stop logic testing
4. ✅ FULL MARKET HOURS: Runs entire trading session (9:15 AM - 3:30 PM IST)
5. ✅ COMPREHENSIVE REPORTING: Detailed logs and performance reports
6. ✅ LIVE TESTING: Tests all logic with real market conditions
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import threading
import queue
from collections import deque
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import logging
from logging.handlers import RotatingFileHandler
from scipy.stats import norm
import asyncio
import json
import signal
from tabulate import tabulate
import colorama
from colorama import Fore, Back, Style
from pathlib import Path
import pytz

# Third-party imports (only KiteConnect for market data)
from kiteconnect import KiteConnect, KiteTicker
import warnings

# Initialize colorama and suppress warnings
colorama.init()
warnings.filterwarnings("ignore")

# Set timezone to IST
IST = pytz.timezone('Asia/Kolkata')

# Create directories
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)
REPORTS_DIR = Path("reports") 
REPORTS_DIR.mkdir(exist_ok=True)

# Environment Variables (from GitHub secrets)
API_KEY = os.getenv("API_KEY", "")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN", "")
TRADING_CAPITAL = float(os.getenv("TRADING_CAPITAL", "10000"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
MAX_DAILY_TRADES_ENV = int(os.getenv("MAX_DAILY_TRADES", "10"))

# Trading Configuration
class TradingConfig:
    LOT_SIZE = 75
    BROKERAGE_PER_LOT = 40
    MIN_PROFIT_AFTER_BROKERAGE = 100
    MAX_POSITION_SIZE = 2
    STRIKE_RANGE_WIDTH = 150
    MIN_CONFIDENCE_THRESHOLD = 0.65
    TRADE_COOLDOWN_SECONDS = 60
    MAX_DAILY_TRADES = MAX_DAILY_TRADES_ENV
    STOP_LOSS_PERCENTAGE = 0.05
    MIN_RISK_REWARD_RATIO = 0.75
    
    # Profit-Based Trailing Configuration
    TRAILING_START_THRESHOLD = 150  # Start trailing when profit > ₹150
    TRAILING_STOP_INCREMENT = 100   # Trail in ₹100 increments
    
    # Capital Management
    MAX_CAPITAL_UTILIZATION = 0.80
    FOCUS_MODE_THRESHOLD = 0.70
    
    # Market Hours (IST)
    MARKET_START_HOUR = 9
    MARKET_START_MINUTE = 15
    MARKET_END_HOUR = 15
    MARKET_END_MINUTE = 30
    
    # Real-time processing
    REALTIME_STOP_LOSS_MONITORING = True
    SIGNAL_ANALYSIS_EVERY_N_TICKS = 5


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors"""
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT,
    }
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{log_color}{record.levelname}{Style.RESET_ALL}"
        return super().format(record)


def setup_logging():
    """Setup comprehensive logging system"""
    console_formatter = ColoredFormatter(
        '%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    
    file_formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, LOG_LEVEL))
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, LOG_LEVEL))
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler
    today = datetime.now(IST).strftime('%Y-%m-%d')
    main_log_file = LOGS_DIR / f"trading_{today}.log"
    
    file_handler = RotatingFileHandler(
        main_log_file, maxBytes=50*1024*1024, backupCount=5, encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # Order recording logger
    order_logger = logging.getLogger('orders')
    order_log_file = LOGS_DIR / f"orders_{today}.log"
    
    order_handler = RotatingFileHandler(
        order_log_file, maxBytes=50*1024*1024, backupCount=10, encoding='utf-8'
    )
    order_handler.setLevel(logging.INFO)
    order_handler.setFormatter(file_formatter)
    order_logger.addHandler(order_handler)
    order_logger.setLevel(logging.INFO)
    
    return logging.getLogger(__name__), order_logger

logger, order_logger = setup_logging()


@dataclass
class RecordedOrder:
    """Structure for recording simulated orders"""
    order_id: str
    timestamp: datetime
    symbol: str
    transaction_type: str  # BUY/SELL
    quantity: int
    order_type: str  # LIMIT/SL/MARKET
    price: float
    trigger_price: Optional[float] = None
    status: str = "PLACED"  # PLACED/EXECUTED/CANCELLED
    execution_price: Optional[float] = None
    execution_time: Optional[datetime] = None
    reason: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class TradeReport:
    """Comprehensive trade report structure"""
    trade_id: str
    symbol: str
    option_type: str
    strike: float
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    quantity: int
    side: str
    
    # P&L Breakdown
    gross_pnl: float
    brokerage: float
    net_pnl: float
    
    # Trade Details
    deployed_capital: float
    max_favorable_price: float
    trailing_activated: bool
    target_hit: bool
    stop_loss_hit: bool
    exit_reason: str
    
    # Performance Metrics
    holding_duration_minutes: float
    roi_percentage: float
    
    # Order tracking
    entry_order_id: str
    exit_order_id: str
    stop_order_ids: List[str]
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class MarketDepth:
    """Market depth structure"""
    timestamp: float
    bids: List[Tuple[float, int, int]]
    asks: List[Tuple[float, int, int]]
    
    @property
    def spread(self) -> float:
        if self.bids and self.asks:
            return self.asks[0][0] - self.bids[0][0]
        return float("inf")
    
    @property
    def mid_price(self) -> float:
        if self.bids and self.asks:
            return (self.bids[0][0] + self.asks[0][0]) / 2
        return 0
    
    @property
    def volume_imbalance(self) -> float:
        bid_volume = sum(b[1] for b in self.bids[:5])
        ask_volume = sum(a[1] for a in self.asks[:5])
        total_volume = bid_volume + ask_volume
        if total_volume > 0:
            return (bid_volume - ask_volume) / total_volume
        return 0
    
    @property
    def total_volume(self) -> int:
        bid_volume = sum(b[1] for b in self.bids[:5])
        ask_volume = sum(a[1] for a in self.asks[:5])
        return bid_volume + ask_volume


class TrendAnalyzer:
    """Trend analysis for market direction"""
    def __init__(self, lookback_periods=50):
        self.price_history = deque(maxlen=lookback_periods)
        self.volume_history = deque(maxlen=lookback_periods)
    
    def update(self, price: float, volume: int):
        self.price_history.append(price)
        self.volume_history.append(volume)
    
    def get_trend_signal(self) -> Dict:
        if len(self.price_history) < 20:
            return {'trend': 'NEUTRAL', 'strength': 0, 'vwap': 0}
        
        prices = np.array(self.price_history)
        volumes = np.array(self.volume_history)
        
        vwap = np.sum(prices * volumes) / np.sum(volumes) if np.sum(volumes) > 0 else prices[-1]
        
        x = np.arange(len(prices))
        slope = np.polyfit(x, prices, 1)[0]
        price_std = np.std(prices)
        strength = abs(slope) / price_std if price_std > 0 else 0
        
        if slope > price_std * 0.01:
            trend = 'BULLISH'
        elif slope < -price_std * 0.01:
            trend = 'BEARISH'
        else:
            trend = 'NEUTRAL'
        
        recent_volume = np.mean(volumes[-5:])
        avg_volume = np.mean(volumes[:-5]) if len(volumes) > 5 else recent_volume
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
        
        if volume_ratio > 1.2:
            strength *= 1.5
        elif volume_ratio < 0.8:
            strength *= 0.5
        
        return {
            'trend': trend,
            'strength': min(1.0, strength),
            'vwap': vwap,
            'current_vs_vwap': (prices[-1] - vwap) / vwap,
            'volume_ratio': volume_ratio,
        }


class Position:
    """Position class with profit-based trailing logic and order recording"""
    
    def __init__(self, symbol: str, side: str, entry_price: float, quantity: int, 
                 entry_time: datetime, option_type: str, strike: float, 
                 expected_profit: float, deployed_capital: float, order_recorder):
        
        self.trade_id = f"{datetime.now(IST).strftime('%Y%m%d')}_{int(time.time())}"
        self.symbol = symbol
        self.side = side
        self.entry_price = entry_price
        self.quantity = quantity
        self.entry_time = entry_time
        self.option_type = option_type
        self.strike = strike
        self.expected_profit = expected_profit
        self.deployed_capital = deployed_capital
        self.order_recorder = order_recorder
        
        # Exit tracking
        self.exit_price = None
        self.exit_time = None
        self.exit_reason = None
        
        # Stop loss and target tracking
        self.stop_loss = None
        self.target = None
        self.initial_stop = None
        self.original_stop = None
        self.original_target = None
        
        # Trailing stop logic
        self.trailing_start_price = None
        self.max_favorable_price = entry_price
        self.is_trailing_stop_active = False
        self.target_removed = False
        self.trailing_stop_price = None
        self.current_trailing_level = 0
        self.trailing_increment = 0
        self.original_stop_reinforced = False
        
        # Order tracking
        self.entry_order_id = None
        self.stop_order_ids = []
        self.exit_order_id = None
        
        logger.info(f"NEW POSITION CREATED: {self.trade_id} - {symbol} {side} {quantity}@{entry_price}")
    
    def update_stops(self, stop_loss: float, target: float):
        """Update stop loss and target"""
        if self.initial_stop is None:
            stop_loss_amount = self.deployed_capital * TradingConfig.STOP_LOSS_PERCENTAGE
            stop_loss_per_share = stop_loss_amount / self.quantity
            self.initial_stop = max(
                self.entry_price - stop_loss_per_share, 
                self.entry_price * 0.95
            )
        
        self.stop_loss = stop_loss if stop_loss else self.initial_stop
        self.target = target
        self.original_stop = self.stop_loss
        self.original_target = self.target
        
        # Setup profit-based trailing
        if self.side == "BUY":
            profit_per_share_for_trailing = TradingConfig.TRAILING_START_THRESHOLD / self.quantity
            self.trailing_start_price = self.entry_price + profit_per_share_for_trailing
            self.trailing_increment = TradingConfig.TRAILING_STOP_INCREMENT / self.quantity
        
        logger.info(f"POSITION {self.trade_id} SETUP:")
        logger.info(f"  Entry: ₹{self.entry_price:.2f}")
        logger.info(f"  Target: ₹{self.target:.2f}")
        logger.info(f"  Stop: ₹{self.stop_loss:.2f}")
        logger.info(f"  Trailing starts at profit > ₹{TradingConfig.TRAILING_START_THRESHOLD}")
    
    def round_price_for_order(self, price: float, is_buy: bool) -> float:
        """Round price to 0.05 multiples"""
        try:
            if not isinstance(price, (int, float)) or price <= 0:
                return 0.05
            
            if is_buy:
                rounded_price = int(price / 0.05) * 0.05
            else:
                rounded_price = int((price + 0.049) / 0.05) * 0.05
            
            return max(0.05, round(rounded_price, 2))
        except Exception as e:
            logger.error(f"Error rounding price {price}: {e}")
            return max(0.05, round(float(price), 2))
    
    def should_start_trailing(self, current_price: float) -> bool:
        """Check if trailing should start"""
        if self.side != "BUY" or not self.trailing_start_price:
            return False
        
        current_profit = (current_price - self.entry_price) * self.quantity
        
        if current_profit >= TradingConfig.TRAILING_START_THRESHOLD:
            if not self.is_trailing_stop_active:
                logger.info(f"{Fore.GREEN}🎯 STARTING PROFIT-BASED TRAILING: {self.trade_id}")
                logger.info(f"   Profit ₹{current_profit:.0f} >= ₹{TradingConfig.TRAILING_START_THRESHOLD}{Style.RESET_ALL}")
            return True
        return False
    
    def update_trailing_stop(self, current_price: float) -> bool:
        """Update trailing stop with profit-based logic"""
        try:
            if self.side != "BUY":
                return False
            
            if not self.should_start_trailing(current_price):
                return False
            
            # Remove target when trailing starts
            if not self.target_removed:
                self.target_removed = True
                self.target = None
                logger.info(f"{Fore.YELLOW}📈 TARGET REMOVED: {self.trade_id} - Letting position run{Style.RESET_ALL}")
            
            # Calculate trailing level
            current_profit = (current_price - self.entry_price) * self.quantity
            trailing_level = max(0, int((current_profit - TradingConfig.TRAILING_START_THRESHOLD) / TradingConfig.TRAILING_STOP_INCREMENT))
            target_profit_for_stop = TradingConfig.TRAILING_START_THRESHOLD + (trailing_level * TradingConfig.TRAILING_STOP_INCREMENT)
            target_stop_price = self.entry_price + (target_profit_for_stop / self.quantity)
            
            # Activate or update trailing
            if not self.is_trailing_stop_active:
                self.is_trailing_stop_active = True
                self.trailing_stop_price = target_stop_price
                self.current_trailing_level = trailing_level
                
                logger.info(f"{Fore.GREEN}🔒 INITIAL TRAILING: {self.trade_id}")
                logger.info(f"   Profit: ₹{current_profit:.0f}, Level: {trailing_level}")
                logger.info(f"   Stop: ₹{target_stop_price:.2f}{Style.RESET_ALL}")
                
                return self._record_trailing_stop_order(target_stop_price)
            
            # Update to next level
            if trailing_level > self.current_trailing_level:
                old_level = self.current_trailing_level
                self.current_trailing_level = trailing_level
                self.trailing_stop_price = target_stop_price
                
                logger.info(f"{Fore.GREEN}🔄 TRAILING UPDATE: {self.trade_id}")
                logger.info(f"   Level: {old_level} → {trailing_level}")
                logger.info(f"   Stop: ₹{target_stop_price:.2f}{Style.RESET_ALL}")
                
                return self._record_trailing_stop_order(target_stop_price)
            
            # Update max favorable price
            if current_price > self.max_favorable_price:
                self.max_favorable_price = current_price
            
            return False
            
        except Exception as e:
            logger.error(f"Error updating trailing stop for {self.trade_id}: {e}")
            return False
    
    def _record_trailing_stop_order(self, stop_price: float) -> bool:
        """Record trailing stop order instead of placing it"""
        try:
            # Cancel previous stop order (record cancellation)
            if self.stop_order_ids:
                last_stop_id = self.stop_order_ids[-1]
                cancel_order = RecordedOrder(
                    order_id=f"CANCEL_{last_stop_id}",
                    timestamp=datetime.now(IST),
                    symbol=self.symbol,
                    transaction_type="CANCEL",
                    quantity=0,
                    order_type="CANCEL",
                    price=0,
                    status="CANCELLED",
                    reason="Replacing with new trailing stop"
                )
                self.order_recorder.record_order(cancel_order)
            
            # Record new stop order
            rounded_stop_price = self.round_price_for_order(stop_price, is_buy=False)
            trigger_price = rounded_stop_price
            limit_price = rounded_stop_price - 0.05
            
            stop_order_id = f"SL_{self.trade_id}_{len(self.stop_order_ids)+1}"
            
            stop_order = RecordedOrder(
                order_id=stop_order_id,
                timestamp=datetime.now(IST),
                symbol=self.symbol,
                transaction_type="SELL",
                quantity=self.quantity,
                order_type="SL",
                price=limit_price,
                trigger_price=trigger_price,
                status="PLACED",
                reason="Trailing stop loss"
            )
            
            self.order_recorder.record_order(stop_order)
            self.stop_order_ids.append(stop_order_id)
            self.stop_loss = rounded_stop_price
            
            logger.info(f"{Fore.GREEN}✅ TRAILING STOP RECORDED: {stop_order_id}{Style.RESET_ALL}")
            return True
            
        except Exception as e:
            logger.error(f"Error recording trailing stop order: {e}")
            return False
    
    def check_emergency_exit(self, current_price: float) -> bool:
        """Check if emergency exit is needed"""
        if self.side == "BUY" and current_price <= self.original_stop:
            logger.error(f"{Fore.RED}🚨 EMERGENCY EXIT: {self.trade_id} - Price ₹{current_price:.2f} <= Original Stop ₹{self.original_stop:.2f}{Style.RESET_ALL}")
            return True
        return False
    
    def close(self, exit_price: float, exit_time: datetime, exit_reason: str = "UNKNOWN") -> TradeReport:
        """Close position and generate trade report"""
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.exit_reason = exit_reason
        
        # Calculate P&L
        if self.side == "BUY":
            gross_pnl = (exit_price - self.entry_price) * self.quantity
        else:
            gross_pnl = (self.entry_price - exit_price) * self.quantity
        
        # Calculate brokerage
        num_lots = self.quantity / TradingConfig.LOT_SIZE
        total_brokerage = TradingConfig.BROKERAGE_PER_LOT * 2 * num_lots
        net_pnl = gross_pnl - total_brokerage
        
        # Calculate metrics
        holding_duration = (exit_time - self.entry_time).total_seconds() / 60
        roi_percentage = (net_pnl / self.deployed_capital) * 100 if self.deployed_capital > 0 else 0
        
        # Record exit order
        exit_order_id = f"EXIT_{self.trade_id}"
        exit_order = RecordedOrder(
            order_id=exit_order_id,
            timestamp=exit_time,
            symbol=self.symbol,
            transaction_type="SELL" if self.side == "BUY" else "BUY",
            quantity=self.quantity,
            order_type="LIMIT",
            price=exit_price,
            status="EXECUTED",
            execution_price=exit_price,
            execution_time=exit_time,
            reason=exit_reason
        )
        self.order_recorder.record_order(exit_order)
        self.exit_order_id = exit_order_id
        
        # Create trade report
        trade_report = TradeReport(
            trade_id=self.trade_id,
            symbol=self.symbol,
            option_type=self.option_type,
            strike=self.strike,
            entry_time=self.entry_time,
            exit_time=exit_time,
            entry_price=self.entry_price,
            exit_price=exit_price,
            quantity=self.quantity,
            side=self.side,
            gross_pnl=gross_pnl,
            brokerage=total_brokerage,
            net_pnl=net_pnl,
            deployed_capital=self.deployed_capital,
            max_favorable_price=self.max_favorable_price,
            trailing_activated=self.is_trailing_stop_active,
            target_hit=(exit_reason == "TARGET_ACHIEVED"),
            stop_loss_hit=(exit_reason in ["STOP_LOSS_HIT", "EMERGENCY_EXIT"]),
            exit_reason=exit_reason,
            holding_duration_minutes=holding_duration,
            roi_percentage=roi_percentage,
            entry_order_id=self.entry_order_id or "",
            exit_order_id=exit_order_id,
            stop_order_ids=self.stop_order_ids
        )
        
        logger.info(f"{Fore.CYAN}TRADE COMPLETED: {self.trade_id}")
        logger.info(f"  Gross P&L: ₹{gross_pnl:.2f}")
        logger.info(f"  Net P&L: ₹{net_pnl:.2f}")
        logger.info(f"  Duration: {holding_duration:.1f} minutes{Style.RESET_ALL}")
        
        return trade_report


class OrderRecorder:
    """Records all orders instead of placing them"""
    
    def __init__(self):
        self.orders: List[RecordedOrder] = []
        self.order_count = 0
    
    def record_order(self, order: RecordedOrder):
        """Record an order"""
        self.orders.append(order)
        self.order_count += 1
        
        # Log to order logger
        order_logger.info(f"ORDER_RECORDED: {json.dumps(order.to_dict(), default=str, indent=2)}")
        
        # Log to main logger
        logger.info(f"📝 ORDER #{self.order_count}: {order.transaction_type} {order.quantity} {order.symbol} @ ₹{order.price:.2f}")
    
    def get_orders_summary(self) -> Dict:
        """Get summary of all recorded orders"""
        entry_orders = [o for o in self.orders if o.transaction_type == "BUY"]
        exit_orders = [o for o in self.orders if o.transaction_type == "SELL" and o.order_type != "SL"]
        stop_orders = [o for o in self.orders if o.order_type == "SL"]
        cancel_orders = [o for o in self.orders if o.transaction_type == "CANCEL"]
        
        return {
            'total_orders': len(self.orders),
            'entry_orders': len(entry_orders),
            'exit_orders': len(exit_orders),
            'stop_orders': len(stop_orders),
            'cancel_orders': len(cancel_orders),
            'orders': [o.to_dict() for o in self.orders]
        }


class TradingMetrics:
    """Trading performance metrics"""
    def __init__(self):
        self.trades: List[TradeReport] = []
        self.daily_pnl = 0
        self.total_brokerage = 0
    
    def add_trade(self, trade: TradeReport):
        self.trades.append(trade)
        self.daily_pnl += trade.net_pnl
        self.total_brokerage += trade.brokerage
    
    def calculate_metrics(self) -> Dict:
        if not self.trades:
            return {
                'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0,
                'win_rate': 0, 'profit_factor': 0, 'avg_winner': 0, 'avg_loser': 0,
                'largest_winner': 0, 'largest_loser': 0, 'net_pnl': 0,
                'gross_pnl': 0, 'total_costs': 0, 'roi': 0
            }
        
        winning_trades = [t for t in self.trades if t.net_pnl > 0]
        losing_trades = [t for t in self.trades if t.net_pnl <= 0]
        
        total_gross_profit = sum(t.gross_pnl for t in winning_trades)
        total_gross_loss = abs(sum(t.gross_pnl for t in losing_trades))
        
        return {
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(self.trades) * 100,
            'profit_factor': total_gross_profit / total_gross_loss if total_gross_loss > 0 else float('inf'),
            'avg_winner': sum(t.net_pnl for t in winning_trades) / len(winning_trades) if winning_trades else 0,
            'avg_loser': sum(t.net_pnl for t in losing_trades) / len(losing_trades) if losing_trades else 0,
            'largest_winner': max((t.net_pnl for t in winning_trades), default=0),
            'largest_loser': min((t.net_pnl for t in losing_trades), default=0),
            'net_pnl': self.daily_pnl,
            'gross_pnl': sum(t.gross_pnl for t in self.trades),
            'total_costs': self.total_brokerage,
            'roi': (self.daily_pnl / TRADING_CAPITAL * 100) if TRADING_CAPITAL > 0 else 0
        }


def is_market_hours() -> bool:
    """Check if current time is within market hours"""
    now = datetime.now(IST)
    market_start = now.replace(
        hour=TradingConfig.MARKET_START_HOUR, 
        minute=TradingConfig.MARKET_START_MINUTE, 
        second=0, microsecond=0
    )
    market_end = now.replace(
        hour=TradingConfig.MARKET_END_HOUR, 
        minute=TradingConfig.MARKET_END_MINUTE, 
        second=0, microsecond=0
    )
    return market_start <= now <= market_end


def is_trading_day() -> bool:
    """Check if today is a trading day"""
    now = datetime.now(IST)
    return now.weekday() < 5  # Monday to Friday


class NIFTYIntradayEngine:
    """Main NIFTY Options Trading Engine - Real-time Simulation Mode"""
    
    def __init__(self, capital: float):
        self.capital = capital
        self.available_capital = capital
        self.deployed_capital = 0
        
        # Initialize Kite connection for market data
        self.kite = None
        if API_KEY and ACCESS_TOKEN:
            self.kite = KiteConnect(api_key=API_KEY)
            self.kite.set_access_token(ACCESS_TOKEN)
        
        # Trading state
        self.positions: Dict[str, Position] = {}
        self.trading_metrics = TradingMetrics()
        self.order_recorder = OrderRecorder()
        self.trend_analyzer = TrendAnalyzer()
        
        # Market data
        self.option_chains = {}
        self.market_depths = {}
        self.spot_price = 0
        self.monitored_options = {}
        
        # Control variables
        self.is_running = False
        self.tick_queue = queue.Queue()
        self.kws = None
        self.tick_count = 0
        self.last_display_time = 0
        
        # Performance tracking
        self.trades_today = 0
        self.last_trade_time = {}
        self.last_signal_time = {}
        self.margin_used = 0
        
        logger.info(f"{Fore.GREEN}NIFTY Engine Initialized - REAL-TIME SIMULATION MODE{Style.RESET_ALL}")
        logger.info(f"Capital: ₹{capital:,.2f}")
        logger.info(f"API Connection: {'✓ Connected' if self.kite else '✗ No API'}")
        logger.info(f"Trailing: Start ₹{TradingConfig.TRAILING_START_THRESHOLD}, Increment ₹{TradingConfig.TRAILING_STOP_INCREMENT}")
        
        print(f"\n{Fore.YELLOW}{'='*80}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}🔧 REAL-TIME SIMULATION MODE 🔧{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Using live market data to test trading logic{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}All orders will be recorded, not executed{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}{'='*80}{Style.RESET_ALL}\n")
    
    def is_in_focus_mode(self) -> bool:
        """Check if in focus mode"""
        capital_utilization = self.margin_used / self.capital if self.capital > 0 else 0
        return capital_utilization >= TradingConfig.FOCUS_MODE_THRESHOLD
    
    def calculate_position_size_enhanced(self, confidence: float, entry_price: float) -> Tuple[int, float]:
        """Calculate position size"""
        try:
            available_for_trading = self.capital * TradingConfig.MAX_CAPITAL_UTILIZATION - self.margin_used
            
            if self.is_in_focus_mode():
                logger.info(f"{Fore.YELLOW}FOCUS MODE: {self.margin_used/self.capital:.1%} capital used{Style.RESET_ALL}")
                return 0, 0
            
            base_quantity = TradingConfig.LOT_SIZE
            required_capital = entry_price * base_quantity
            
            if required_capital > available_for_trading:
                logger.warning(f"{Fore.YELLOW}INSUFFICIENT CAPITAL: Need ₹{required_capital:,.2f}, Available ₹{available_for_trading:,.2f}{Style.RESET_ALL}")
                return 0, 0
            
            confidence_factor = min(1.0, confidence * 1.2)
            max_lots = min(2, int(confidence_factor * 2) + 1)
            max_affordable_lots = int(available_for_trading / (entry_price * TradingConfig.LOT_SIZE))
            
            final_lots = min(max_lots, max_affordable_lots, TradingConfig.MAX_POSITION_SIZE)
            final_quantity = final_lots * TradingConfig.LOT_SIZE
            final_capital_required = entry_price * final_quantity
            
            logger.info(f"POSITION SIZING: {final_lots} lots, ₹{final_capital_required:,.2f} required")
            return final_quantity, final_capital_required
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return TradingConfig.LOT_SIZE, entry_price * TradingConfig.LOT_SIZE
    
    def round_price_for_order(self, price: float, is_buy: bool) -> float:
        """Round price to valid increments"""
        try:
            if is_buy:
                rounded = int(price / 0.05) * 0.05
            else:
                rounded = int((price + 0.04) / 0.05) * 0.05
            return round(rounded, 2)
        except Exception as e:
            logger.error(f"Error rounding price {price}: {e}")
            return round(price, 2)
    
    def calculate_expected_profit(self, entry_price: float, quantity: int, 
                                target_price: float, stop_price: float) -> Dict:
        """Calculate expected profit with charges"""
        num_lots = quantity / TradingConfig.LOT_SIZE
        
        profit_per_lot = abs(target_price - entry_price) * TradingConfig.LOT_SIZE
        loss_per_lot = abs(entry_price - stop_price) * TradingConfig.LOT_SIZE
        brokerage_per_lot = TradingConfig.BROKERAGE_PER_LOT * 2
        
        net_profit_per_lot = profit_per_lot - brokerage_per_lot
        net_loss_per_lot = loss_per_lot + brokerage_per_lot
        
        total_net_profit_potential = net_profit_per_lot * num_lots
        total_net_loss_potential = net_loss_per_lot * num_lots
        
        risk_reward = net_profit_per_lot / net_loss_per_lot if net_loss_per_lot > 0 else 0
        
        min_price_movement_required = (TradingConfig.MIN_PROFIT_AFTER_BROKERAGE + brokerage_per_lot) / TradingConfig.LOT_SIZE
        actual_price_movement = abs(target_price - entry_price)
        
        is_worth_trading = (
            net_profit_per_lot >= TradingConfig.MIN_PROFIT_AFTER_BROKERAGE and
            risk_reward >= TradingConfig.MIN_RISK_REWARD_RATIO and
            actual_price_movement >= min_price_movement_required
        )
        
        return {
            'net_profit_per_lot': net_profit_per_lot,
            'net_loss_per_lot': net_loss_per_lot,
            'total_net_profit_potential': total_net_profit_potential,
            'total_net_loss_potential': total_net_loss_potential,
            'risk_reward_ratio': risk_reward,
            'brokerage_per_lot': brokerage_per_lot,
            'min_price_movement_required': min_price_movement_required,
            'actual_price_movement': actual_price_movement,
            'is_worth_trading': is_worth_trading,
        }
    
    def get_tradeable_strikes(self) -> List[Dict]:
        """Get tradeable option strikes using live data"""
        try:
            if not self.kite:
                logger.error("Kite connection not available")
                return []
            
            # Get NIFTY spot price
            nifty_quote = self.kite.quote(["NSE:NIFTY 50"])
            self.spot_price = nifty_quote["NSE:NIFTY 50"]["last_price"]
            
            logger.info(f"{Fore.CYAN}NIFTY Spot Price: ₹{self.spot_price:.2f}{Style.RESET_ALL}")
            
            # Update trend analyzer
            self.trend_analyzer.update(self.spot_price, 1000)
            
            # Calculate strike range
            atm_strike = round(self.spot_price / 50) * 50
            lower_strike = atm_strike - TradingConfig.STRIKE_RANGE_WIDTH
            upper_strike = atm_strike + TradingConfig.STRIKE_RANGE_WIDTH
            
            # Get instruments
            instruments = pd.DataFrame(self.kite.instruments("NFO"))
            
            # Filter NIFTY options
            nifty_options = instruments[
                (instruments['name'] == 'NIFTY') &
                (instruments['strike'] >= lower_strike) &
                (instruments['strike'] <= upper_strike)
            ]
            
            # Find nearest weekly expiry
            nifty_options['expiry_dt'] = pd.to_datetime(nifty_options['expiry'])
            current_date = datetime.now().date()
            
            future_expiries = nifty_options[
                nifty_options['expiry_dt'].dt.date >= current_date
            ]['expiry_dt'].unique()
            
            sorted_expiries = sorted(future_expiries)
            
            # Select weekly expiry (within 7 days)
            weekly_expiry = None
            for exp in sorted_expiries:
                days_to_exp = (exp.date() - current_date).days
                if days_to_exp <= 7:
                    weekly_expiry = exp
                    break
            
            if weekly_expiry is None:
                weekly_expiry = sorted_expiries[0]
            
            logger.info(f"{Fore.CYAN}Selected Expiry: {weekly_expiry.strftime('%d-%b-%Y')} ({(weekly_expiry.date() - current_date).days} days){Style.RESET_ALL}")
            
            # Filter by selected expiry
            tradeable_options = nifty_options[nifty_options['expiry_dt'] == weekly_expiry]
            tradeable_options = tradeable_options.sort_values(['strike', 'instrument_type'])
            
            return tradeable_options.to_dict('records')
            
        except Exception as e:
            logger.error(f"Error getting tradeable strikes: {e}")
            return []
    
    async def execute_trade(self, symbol: str, token: int, side: str, quantity: int, 
                          price: float, target_price: float, stop_price: float, 
                          option_info: Dict, expected_profit: float, deployed_capital: float):
        """Execute trade by recording orders"""
        try:
            # Record entry order
            entry_order_id = f"ENTRY_{self.trades_today + 1}_{int(time.time())}"
            entry_order = RecordedOrder(
                order_id=entry_order_id,
                timestamp=datetime.now(IST),
                symbol=symbol,
                transaction_type=side,
                quantity=quantity,
                order_type="LIMIT",
                price=price,
                status="EXECUTED",
                execution_price=price,
                execution_time=datetime.now(IST),
                reason="Entry order"
            )
            self.order_recorder.record_order(entry_order)
            
            # Create position
            position = Position(
                symbol=symbol,
                side=side,
                entry_price=price,
                quantity=quantity,
                entry_time=datetime.now(IST),
                option_type=option_info['type'],
                strike=option_info['strike'],
                expected_profit=expected_profit,
                deployed_capital=deployed_capital,
                order_recorder=self.order_recorder
            )
            
            position.entry_order_id = entry_order_id
            position.update_stops(stop_price, target_price)
            
            # Record initial stop loss order
            trigger_price = position.round_price_for_order(stop_price, is_buy=False)
            limit_price = trigger_price - 0.05
            
            stop_order_id = f"SL_{position.trade_id}_INITIAL"
            stop_order = RecordedOrder(
                order_id=stop_order_id,
                timestamp=datetime.now(IST),
                symbol=symbol,
                transaction_type="SELL",
                quantity=quantity,
                order_type="SL",
                price=limit_price,
                trigger_price=trigger_price,
                status="PLACED",
                reason="Initial stop loss"
            )
            self.order_recorder.record_order(stop_order)
            position.stop_order_ids.append(stop_order_id)
            
            logger.info(f"{Fore.GREEN}TRADE RECORDED:{Style.RESET_ALL}")
            logger.info(f"Symbol: {symbol}")
            logger.info(f"Entry: ₹{price:.2f} | Target: ₹{target_price:.2f} | Stop: ₹{stop_price:.2f}")
            logger.info(f"Quantity: {quantity} | Capital: ₹{deployed_capital:,.2f}")
            logger.info(f"Orders: Entry {entry_order_id}, Stop {stop_order_id}")
            
            # Add to active positions
            self.positions[token] = position
            self.last_trade_time[token] = datetime.now()
            self.trades_today += 1
            
            # Update capital tracking
            self.deployed_capital += deployed_capital
            self.margin_used += deployed_capital
            
            logger.info(f"{Fore.GREEN}✅ TRADE SETUP COMPLETE: {symbol}{Style.RESET_ALL}")
            
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
    
    async def close_position(self, token: int, current_price: float, exit_reason: str = "UNKNOWN"):
        """Close position and generate report"""
        try:
            position = self.positions[token]
            
            logger.info(f"{Fore.YELLOW}CLOSING POSITION: {position.trade_id} - Reason: {exit_reason}{Style.RESET_ALL}")
            
            # Generate trade report
            trade_report = position.close(current_price, datetime.now(IST), exit_reason)
            
            # Update metrics
            self.trading_metrics.add_trade(trade_report)
            
            # Update capital tracking
            self.deployed_capital -= position.deployed_capital
            self.margin_used -= position.deployed_capital
            
            # Remove from active positions
            del self.positions[token]
            
            # Log performance update
            metrics = self.trading_metrics.calculate_metrics()
            logger.info(f"{Fore.CYAN}PERFORMANCE UPDATE:{Style.RESET_ALL}")
            logger.info(f"  Net P&L: ₹{metrics['net_pnl']:.2f}")
            logger.info(f"  Win Rate: {metrics['win_rate']:.1f}%")
            logger.info(f"  Total Trades: {metrics['total_trades']}")
            
        except Exception as e:
            logger.error(f"Position close failed: {e}")
    
    async def process_stop_losses_realtime(self, tick: Dict):
        """Real-time stop loss monitoring"""
        try:
            token = tick.get('instrument_token')
            
            if token not in self.positions:
                return
            
            if 'depth' in tick and tick['depth']['buy'] and tick['depth']['sell']:
                depth_data = tick['depth']
                depth = MarketDepth(
                    timestamp=time.time(),
                    bids=[(d['price'], d['quantity'], d['orders']) 
                          for d in depth_data.get('buy', [])[:5] if d.get('price', 0) > 0],
                    asks=[(d['price'], d['quantity'], d['orders']) 
                          for d in depth_data.get('sell', [])[:5] if d.get('price', 0) > 0],
                )
                
                if depth.bids and depth.asks:
                    current_price = depth.mid_price
                    position = self.positions[token]
                    
                    # Check emergency exit
                    if position.check_emergency_exit(current_price):
                        logger.error(f"{Fore.RED}🚨 EMERGENCY EXIT: {position.trade_id}{Style.RESET_ALL}")
                        await self.close_position(token, current_price, exit_reason="EMERGENCY_EXIT")
                        return
                    
                    # Update trailing stop
                    position.update_trailing_stop(current_price)
                    
                    # Check normal exit conditions
                    exit_reason = None
                    should_exit = False
                    
                    if position.side == "BUY":
                        if current_price <= position.stop_loss:
                            exit_reason = "STOP_LOSS_HIT"
                            should_exit = True
                        elif position.target and current_price >= position.target:
                            exit_reason = "TARGET_ACHIEVED"
                            should_exit = True
                    
                    if should_exit:
                        logger.info(f"{Fore.RED}⚡ REAL-TIME EXIT: {exit_reason} for {position.trade_id}{Style.RESET_ALL}")
                        exit_price_rounded = self.round_price_for_order(current_price, is_buy=False)
                        await self.close_position(token, exit_price_rounded, exit_reason=exit_reason)
                        
        except Exception as e:
            logger.error(f"Error in real-time stop loss processing: {e}")
    
    async def process_entry_signals(self, tick: Dict):
        """Process entry signals periodically"""
        try:
            token = tick.get('instrument_token')
            
            if (token not in self.monitored_options or 
                not is_market_hours()):
                return
            
            if self.is_in_focus_mode():
                return
            
            if 'depth' in tick and tick['depth']['buy'] and tick['depth']['sell']:
                depth_data = tick['depth']
                depth = MarketDepth(
                    timestamp=time.time(),
                    bids=[(d['price'], d['quantity'], d['orders']) 
                          for d in depth_data.get('buy', [])[:5] if d.get('price', 0) > 0],
                    asks=[(d['price'], d['quantity'], d['orders']) 
                          for d in depth_data.get('sell', [])[:5] if d.get('price', 0) > 0],
                )
                
                if depth.bids and depth.asks:
                    self.market_depths[token] = depth
                    await self.check_entry_signals_only(token, depth)
                    
        except Exception as e:
            logger.error(f"Error processing entry signals: {e}")
    
    async def check_entry_signals_only(self, token: int, depth: MarketDepth):
        """Check for entry signals"""
        try:
            option_symbol = self.monitored_options[token]['symbol']
            last_trade = self.last_trade_time.get(token, datetime.min)
            last_signal = self.last_signal_time.get(token, datetime.min)
            
            signal_cooldown = timedelta(seconds=10)
            trade_cooldown = timedelta(seconds=TradingConfig.TRADE_COOLDOWN_SECONDS)
            now = datetime.now()
            
            if now - last_signal < signal_cooldown:
                return
            
            if now - last_trade < trade_cooldown:
                self.last_signal_time[token] = now
                return
            
            # Get trend signal
            trend_signal = self.trend_analyzer.get_trend_signal()
            if not trend_signal:
                trend_signal = {
                    'trend': 'NEUTRAL', 'strength': 0.1, 'vwap': depth.mid_price,
                    'current_vs_vwap': 0, 'volume_ratio': 1.0,
                }
            
            current_price = depth.mid_price
            
            # Entry conditions
            if (len(self.positions) < TradingConfig.MAX_POSITION_SIZE and
                self.trades_today < TradingConfig.MAX_DAILY_TRADES and
                token not in self.positions and
                depth.spread <= current_price * 0.02 and
                depth.total_volume >= 100 and
                not self.is_in_focus_mode()):
                
                option_info = self.monitored_options[token]
                entry_signal = None
                confidence = 0
                
                # Strong buy signal - High volume imbalance
                if depth.volume_imbalance > 0.35:
                    entry_signal = "BUY"
                    confidence = min(0.85, 0.65 + abs(depth.volume_imbalance) * 0.4)
                    logger.info(f"{Fore.GREEN}STRONG BUY SIGNAL: Volume imbalance {depth.volume_imbalance:.1%}{Style.RESET_ALL}")
                
                # Trend buy signal
                elif (depth.volume_imbalance > 0.25 and
                      trend_signal['trend'] == 'BULLISH' and
                      trend_signal['strength'] >= 0.4 and
                      trend_signal['volume_ratio'] > 1.3):
                    entry_signal = "BUY"
                    confidence = min(0.80, 0.60 + trend_signal['strength'] * 0.3)
                    logger.info(f"{Fore.CYAN}TREND BUY SIGNAL: Trend {trend_signal['strength']:.2f}, Imbalance {depth.volume_imbalance:.1%}{Style.RESET_ALL}")
                
                # Execute trade if signal meets criteria
                if entry_signal == "BUY" and confidence >= TradingConfig.MIN_CONFIDENCE_THRESHOLD:
                    
                    entry_price_rounded = self.round_price_for_order(current_price, is_buy=True)
                    position_size, required_capital = self.calculate_position_size_enhanced(confidence, entry_price_rounded)
                    
                    if position_size > 0:
                        
                        # Calculate target and stop prices
                        target_price = current_price * 1.06
                        target_price_rounded = self.round_price_for_order(target_price, is_buy=False)
                        
                        stop_loss_amount = required_capital * TradingConfig.STOP_LOSS_PERCENTAGE
                        stop_loss_per_share = stop_loss_amount / position_size
                        stop_price_calc = max(
                            entry_price_rounded - stop_loss_per_share,
                            entry_price_rounded * 0.95
                        )
                        stop_price_rounded = self.round_price_for_order(stop_price_calc, is_buy=False)
                        
                        # Analyze trade profitability
                        trade_analysis = self.calculate_expected_profit(
                            entry_price_rounded, position_size, target_price_rounded, stop_price_rounded
                        )
                        
                        logger.info(f"{Fore.CYAN}TRADE ANALYSIS:{Style.RESET_ALL}")
                        logger.info(f"  Net Profit Potential: ₹{trade_analysis['total_net_profit_potential']:.2f}")
                        logger.info(f"  Risk/Reward Ratio: {trade_analysis['risk_reward_ratio']:.2f}")
                        logger.info(f"  Worth Trading: {trade_analysis['is_worth_trading']}")
                        
                        if trade_analysis['is_worth_trading']:
                            logger.info(f"{Fore.GREEN}✅ TRADE APPROVED{Style.RESET_ALL}")
                            await self.execute_trade(
                                symbol=option_symbol,
                                token=token,
                                side="BUY",
                                quantity=position_size,
                                price=entry_price_rounded,
                                target_price=target_price_rounded,
                                stop_price=stop_price_rounded,
                                option_info=option_info,
                                expected_profit=trade_analysis['total_net_profit_potential'],
                                deployed_capital=required_capital,
                            )
                        else:
                            logger.info(f"{Fore.RED}❌ TRADE REJECTED: Not profitable enough{Style.RESET_ALL}")
                
                self.last_signal_time[token] = datetime.now()
                
        except Exception as e:
            logger.error(f"Error in entry signal check: {e}")
    
    async def stream_market_data(self):
        """Stream market data and process ticks"""
        try:
            # Get tradeable options
            tradeable_options = self.get_tradeable_strikes()
            
            if not tradeable_options:
                logger.warning("No tradeable options found")
                return
            
            # Select options around ATM
            atm_strike = round(self.spot_price / 50) * 50
            selected_options = []
            for opt in tradeable_options:
                if abs(opt['strike'] - atm_strike) <= 150:
                    selected_options.append(opt)
            
            selected_options = selected_options[:10]  # Limit to 10 options
            
            # Setup monitoring
            for opt in selected_options:
                self.monitored_options[opt['instrument_token']] = {
                    'symbol': opt['tradingsymbol'],
                    'strike': opt['strike'],
                    'type': opt['instrument_type'],
                    'expiry': opt['expiry_dt'],
                    'lot_size': opt['lot_size'],
                }
            
            tokens = [opt['instrument_token'] for opt in selected_options]
            
            logger.info(f"Monitoring {len(tokens)} options for real-time simulation")
            
            if not self.kite:
                logger.error("Kite connection not available for streaming")
                return
            
            # Setup WebSocket
            self.kws = KiteTicker(API_KEY, ACCESS_TOKEN)
            
            def on_ticks(ws, ticks):
                for tick in ticks:
                    self.tick_queue.put(tick)
            
            def on_connect(ws, response):
                ws.subscribe(tokens)
                ws.set_mode(ws.MODE_FULL, tokens)
                logger.info(f"{Fore.GREEN}Connected to real-time market data stream{Style.RESET_ALL}")
            
            def on_error(ws, code, reason):
                logger.error(f"WebSocket error: {code} - {reason}")
            
            self.kws.on_ticks = on_ticks
            self.kws.on_connect = on_connect
            self.kws.on_error = on_error
            
            # Start WebSocket in separate thread
            ws_thread = threading.Thread(target=self._run_websocket, daemon=True)
            ws_thread.start()
            
            # Process tick queue
            await self._process_tick_queue()
            
        except Exception as e:
            logger.error(f"Error in stream_market_data: {e}")
    
    def _run_websocket(self):
        """Run WebSocket in separate thread"""
        try:
            self.kws.connect(threaded=True)
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
    
    async def _process_tick_queue(self):
        """Process incoming market ticks"""
        tick_count = 0
        while self.is_running:
            try:
                tick = self.tick_queue.get(timeout=0.1)
                tick_count += 1
                self.tick_count = tick_count
                
                # Real-time stop loss monitoring on EVERY tick
                if TradingConfig.REALTIME_STOP_LOSS_MONITORING:
                    await self.process_stop_losses_realtime(tick)
                
                # Entry signal analysis every N ticks
                if tick_count % TradingConfig.SIGNAL_ANALYSIS_EVERY_N_TICKS == 0:
                    await self.process_entry_signals(tick)
                
                # Display updates
                if tick_count % 100 == 0:
                    self.display_monitored_options()
                    
            except queue.Empty:
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Error processing tick: {e}")
                await asyncio.sleep(0.1)
    
    def display_monitored_options(self):
        """Display monitored options status"""
        if not self.monitored_options:
            return
        
        current_time = time.time()
        if current_time - self.last_display_time < 10:  # Update every 10 seconds
            return
        
        self.last_display_time = current_time
        
        focus_mode = self.is_in_focus_mode()
        capital_utilization = self.margin_used / self.capital if self.capital > 0 else 0
        
        print(f"\n{Fore.YELLOW}{'='*80}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}REAL-TIME SIMULATION - {datetime.now(IST).strftime('%H:%M:%S')}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}{'='*80}{Style.RESET_ALL}")
        
        table_data = []
        for token, info in list(self.monitored_options.items())[:10]:
            symbol = info['symbol']
            strike = info['strike']
            option_type = info['type']
            expiry = info['expiry'].strftime('%d-%b')
            
            if token in self.market_depths:
                current_price = self.market_depths[token].mid_price
                spread = self.market_depths[token].spread
                imbalance = self.market_depths[token].volume_imbalance
                volume = self.market_depths[token].total_volume
            else:
                current_price = spread = imbalance = volume = 0
            
            # Position status
            if token in self.positions:
                position = self.positions[token]
                current_profit = (current_price - position.entry_price) * position.quantity
                status_parts = [f"📈 LONG (₹{current_profit:.0f})"]
                
                if position.is_trailing_stop_active:
                    status_parts.append("🔒 TRAIL")
                if position.target_removed:
                    status_parts.append("🎯 NO-TARGET")
                
                position_status = " ".join(status_parts)
            else:
                position_status = "⚪ MONITORING"
            
            table_data.append([
                f"{strike} {option_type}",
                expiry,
                f"₹{current_price:.2f}",
                f"₹{spread:.2f}",
                f"{imbalance:+.1%}",
                f"{volume:,}",
                position_status,
            ])
        
        headers = ["Strike/Type", "Expiry", "Price", "Spread", "Imbalance", "Volume", "Status"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        # Performance summary
        metrics = self.trading_metrics.calculate_metrics()
        orders_summary = self.order_recorder.get_orders_summary()
        
        print(f"{Fore.CYAN}Capital: ₹{self.capital + metrics['net_pnl']:,.2f} | P&L: ₹{metrics['net_pnl']:,.2f} | Win Rate: {metrics['win_rate']:.1f}%{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Positions: {len(self.positions)} | Trades: {metrics['total_trades']} | Orders Recorded: {orders_summary['total_orders']}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Ticks Processed: {self.tick_count:,} | Focus Mode: {'YES' if focus_mode else 'NO'}{Style.RESET_ALL}")
    
    async def generate_daily_report(self):
        """Generate comprehensive daily report"""
        try:
            metrics = self.trading_metrics.calculate_metrics()
            orders_summary = self.order_recorder.get_orders_summary()
            today = datetime.now(IST).strftime('%Y-%m-%d')
            
            # Create report content
            report_content = f"""
# NIFTY Options Trading Simulation Report
Date: {today}
Generated: {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S IST')}
Mode: Real-time Simulation using Live Market Data

## Trading Summary
- **Total Trades**: {metrics['total_trades']}
- **Winning Trades**: {metrics['winning_trades']}
- **Losing Trades**: {metrics['losing_trades']}
- **Win Rate**: {metrics['win_rate']:.2f}%
- **Profit Factor**: {metrics['profit_factor']:.2f}

## Financial Performance (Simulated)
- **Net P&L**: ₹{metrics['net_pnl']:,.2f}
- **Gross P&L**: ₹{metrics['gross_pnl']:,.2f}
- **Total Costs**: ₹{metrics['total_costs']:,.2f}
- **ROI**: {metrics['roi']:.2f}%
- **Average Winner**: ₹{metrics['avg_winner']:,.2f}
- **Average Loser**: ₹{metrics['avg_loser']:,.2f}
- **Largest Winner**: ₹{metrics['largest_winner']:,.2f}
- **Largest Loser**: ₹{metrics['largest_loser']:,.2f}

## Order Summary
- **Total Orders Recorded**: {orders_summary['total_orders']}
- **Entry Orders**: {orders_summary['entry_orders']}
- **Exit Orders**: {orders_summary['exit_orders']}
- **Stop Loss Orders**: {orders_summary['stop_orders']}
- **Cancel Orders**: {orders_summary['cancel_orders']}

## Trading Configuration
- **Capital**: ₹{self.capital:,.2f}
- **Mode**: Real-time Simulation
- **Trailing Start**: ₹{TradingConfig.TRAILING_START_THRESHOLD}
- **Trailing Increment**: ₹{TradingConfig.TRAILING_STOP_INCREMENT}
- **Max Capital Utilization**: {TradingConfig.MAX_CAPITAL_UTILIZATION:.1%}
- **Stop Loss**: {TradingConfig.STOP_LOSS_PERCENTAGE:.1%} of deployed capital

## Market Data Source
- **Live Data**: KiteConnect API
- **Ticks Processed**: {self.tick_count:,}
- **Options Monitored**: {len(self.monitored_options)}

## Individual Trades
"""
            
            # Add individual trade details
            for i, trade in enumerate(self.trading_metrics.trades, 1):
                report_content += f"""
### Trade {i}: {trade.symbol}
- **Trade ID**: {trade.trade_id}
- **Type**: {trade.option_type} {trade.strike}
- **Entry**: {trade.entry_time.strftime('%H:%M:%S')} @ ₹{trade.entry_price:.2f}
- **Exit**: {trade.exit_time.strftime('%H:%M:%S')} @ ₹{trade.exit_price:.2f}
- **Quantity**: {trade.quantity}
- **Deployed Capital**: ₹{trade.deployed_capital:,.2f}
- **Gross P&L**: ₹{trade.gross_pnl:.2f}
- **Net P&L**: ₹{trade.net_pnl:.2f}
- **ROI**: {trade.roi_percentage:.2f}%
- **Duration**: {trade.holding_duration_minutes:.1f} minutes
- **Exit Reason**: {trade.exit_reason}
- **Trailing Used**: {'Yes' if trade.trailing_activated else 'No'}
- **Entry Order**: {trade.entry_order_id}
- **Exit Order**: {trade.exit_order_id}
- **Stop Orders**: {', '.join(trade.stop_order_ids)}
"""
            
            # Save report
            report_file = REPORTS_DIR / f"simulation_report_{today}.md"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            logger.info(f"{Fore.GREEN}Simulation report generated: {report_file}{Style.RESET_ALL}")
            
            # Also save as JSON
            json_report = {
                'date': today,
                'generated': datetime.now(IST).isoformat(),
                'mode': 'real_time_simulation',
                'metrics': metrics,
                'orders_summary': orders_summary,
                'trades': [trade.to_dict() for trade in self.trading_metrics.trades],
                'config': {
                    'capital': self.capital,
                    'trailing_start': TradingConfig.TRAILING_START_THRESHOLD,
                    'trailing_increment': TradingConfig.TRAILING_STOP_INCREMENT,
                },
                'market_data': {
                    'ticks_processed': self.tick_count,
                    'options_monitored': len(self.monitored_options),
                    'api_used': 'KiteConnect'
                }
            }
            
            json_file = REPORTS_DIR / f"simulation_report_{today}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(json_report, f, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"Error generating daily report: {e}")
    
    async def performance_monitor(self):
        """Monitor performance during trading session"""
        while self.is_running:
            try:
                await asyncio.sleep(300)  # 5 minutes
                
                metrics = self.trading_metrics.calculate_metrics()
                orders_summary = self.order_recorder.get_orders_summary()
                current_capital = self.capital + metrics['net_pnl']
                returns = metrics['roi']
                
                print(f"\n{Fore.CYAN}=== SIMULATION PERFORMANCE UPDATE ==={Style.RESET_ALL}")
                print(f"Capital: ₹{current_capital:,.2f} ({returns:+.2f}%)")
                print(f"P&L: ₹{metrics['net_pnl']:,.2f}")
                print(f"Positions: {len(self.positions)}")
                print(f"Orders Recorded: {orders_summary['total_orders']}")
                print(f"Win Rate: {metrics['win_rate']:.1f}%")
                print(f"Ticks Processed: {self.tick_count:,}")
                
            except Exception as e:
                logger.error(f"Performance monitor error: {e}")
    
    async def start(self):
        """Start the trading engine"""
        try:
            if not API_KEY or not ACCESS_TOKEN:
                logger.error("API credentials not available. Cannot use live market data.")
                return
            
            self.is_running = True
            logger.info("Starting Real-time Simulation Trading Engine...")
            
            if is_market_hours():
                logger.info("📈 Market is open - running live simulation")
            else:
                logger.info("🕐 Market is closed - will wait for market hours or exit after demo")
            
            tasks = [
                self.stream_market_data(),
                self.performance_monitor(),
            ]
            
            await asyncio.gather(*tasks)
            
        except KeyboardInterrupt:
            await self.shutdown()
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down trading engine...")
        self.is_running = False
        
        if self.kws:
            self.kws.close()
        
        # Close all positions
        for token in list(self.positions.keys()):
            if token in self.market_depths:
                current_price = self.market_depths[token].mid_price
                await self.close_position(token, current_price, exit_reason="SYSTEM_SHUTDOWN")
        
        # Generate final report
        await self.generate_daily_report()
        
        # Final summary
        metrics = self.trading_metrics.calculate_metrics()
        orders_summary = self.order_recorder.get_orders_summary()
        
        print(f"\n{Fore.YELLOW}=== FINAL SIMULATION SUMMARY ==={Style.RESET_ALL}")
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Win Rate: {metrics['win_rate']:.1f}%")
        print(f"Net P&L: ₹{metrics['net_pnl']:,.2f}")
        print(f"ROI: {metrics['roi']:.2f}%")
        print(f"Orders Recorded: {orders_summary['total_orders']}")
        print(f"Ticks Processed: {self.tick_count:,}")
        
        logger.info("Trading engine shutdown complete")


def signal_handler(signum, frame):
    """Handle termination signals"""
    logger.info("Received termination signal, shutting down...")
    sys.exit(0)


async def main_trading_loop():
    """Main trading loop"""
    try:
        # Display startup banner
        print(f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════╗
║         NIFTY OPTIONS REAL-TIME SIMULATION               ║
║          Live Market Data + Order Recording              ║
║          ₹{TradingConfig.TRAILING_START_THRESHOLD} Start | ₹{TradingConfig.TRAILING_STOP_INCREMENT} Increments                     ║
╚══════════════════════════════════════════════════════════╝{Style.RESET_ALL}
        """)
        
        logger.info(f"Trading Capital: ₹{TRADING_CAPITAL:,.2f}")
        logger.info(f"Mode: REAL-TIME SIMULATION")
        logger.info(f"Market Hours: {TradingConfig.MARKET_START_HOUR}:{TradingConfig.MARKET_START_MINUTE:02d} - {TradingConfig.MARKET_END_HOUR}:{TradingConfig.MARKET_END_MINUTE:02d} IST")
        
        # Create and start engine
        engine = NIFTYIntradayEngine(TRADING_CAPITAL)
        await engine.start()
        
    except Exception as e:
        logger.error(f"Error in main trading loop: {e}")
    finally:
        logger.info("Trading session ended")


def main():
    """Main entry point"""
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print(f"""
{Fore.GREEN}NIFTY Options Real-time Simulation System{Style.RESET_ALL}
{'='*50}

Environment Configuration:
- API Key: {'✓ Set' if API_KEY else '✗ Missing'}
- Access Token: {'✓ Set' if ACCESS_TOKEN else '✗ Missing'}
- Trading Capital: ₹{TRADING_CAPITAL:,.2f}
- Mode: Real-time Simulation
- Log Level: {LOG_LEVEL}

Features:
- Live market data from KiteConnect API
- Order recording instead of execution
- Full profit-based trailing logic testing
- Comprehensive reporting

Market Hours: {TradingConfig.MARKET_START_HOUR}:{TradingConfig.MARKET_START_MINUTE:02d} - {TradingConfig.MARKET_END_HOUR}:{TradingConfig.MARKET_END_MINUTE:02d} IST
    """)
    
    try:
        logger.info("Starting real-time simulation immediately...")
        asyncio.run(main_trading_loop())
                
    except KeyboardInterrupt:
        logger.info("Program interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
