#!/usr/bin/env python3
"""
NIFTY Options Intraday Trading System with Profit-Based Trailing Stops
GitHub: https://github.com/yourusername/nifty-options-trader

Features:
- Automated market hours trading (9:15 AM - 3:30 PM IST)
- Profit-based trailing stops (₹150 start, ₹100 increments)
- Comprehensive logging and trade reporting
- Real-time stop loss monitoring
- Conservative capital management
- Detailed P&L tracking with brokerage deduction

Author: Your Name
Version: 2.0.0
License: MIT
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import threading
import queue
import asyncio
import aiohttp
import json
import signal
import logging
from logging.handlers import RotatingFileHandler
from collections import deque
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import pytz
import schedule

# Third-party imports
from scipy.stats import norm, skewnorm
from scipy.integrate import quad
from kiteconnect import KiteConnect, KiteTicker
from tabulate import tabulate
import colorama
from colorama import Fore, Back, Style
import warnings

# Initialize colorama and suppress warnings
colorama.init()
warnings.filterwarnings("ignore")

# Set timezone to IST
IST = pytz.timezone('Asia/Kolkata')

# Create logs directory
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

# Create reports directory
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

# Environment Variables with defaults
API_KEY = os.getenv("API_KEY", "")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN", "")
TRADING_CAPITAL = float(os.getenv("TRADING_CAPITAL", "10000"))  # Default ₹1,00,000
LIVE_TRADING_MODE = os.getenv("LIVE_TRADING_MODE", "false").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Trading Configuration
class TradingConfig:
    """Trading configuration constants"""
    
    # Basic Settings
    LOT_SIZE = 75
    BROKERAGE_PER_LOT = 40  # Per transaction (entry OR exit)
    STT_RATE = 0.0001  # 0.01% STT on options
    EXCHANGE_CHARGES_RATE = 0.0000345  # 0.00345% on turnover
    GST_RATE = 0.18  # 18% GST on brokerage + charges
    SEBI_CHARGES_RATE = 0.000001  # ₹1 per crore turnover
    
    # Trading Limits
    MIN_PROFIT_AFTER_BROKERAGE = 100
    MAX_POSITION_SIZE = 2
    STRIKE_RANGE_WIDTH = 150
    MIN_CONFIDENCE_THRESHOLD = 0.65
    TRADE_COOLDOWN_SECONDS = 60
    MAX_DAILY_TRADES = 10
    STOP_LOSS_PERCENTAGE = 0.05  # 5% of deployed capital
    MIN_RISK_REWARD_RATIO = 0.75
    
    # Profit-Based Trailing Configuration
    TRAILING_START_THRESHOLD = 150  # Start trailing when profit > ₹150
    TRAILING_STOP_INCREMENT = 100   # Trail in ₹100 increments
    
    # Capital Management
    MAX_CAPITAL_UTILIZATION = 0.80  # Use max 80% of capital
    FOCUS_MODE_THRESHOLD = 0.70     # Focus on existing trades at 70%
    
    # Processing Configuration
    REALTIME_STOP_LOSS_MONITORING = True
    SIGNAL_ANALYSIS_EVERY_N_TICKS = 5
    UPDATE_FREQUENCY_SECONDS = 10
    
    # Market Hours (IST)
    MARKET_START_HOUR = 9
    MARKET_START_MINUTE = 15
    MARKET_END_HOUR = 15
    MARKET_END_MINUTE = 30
    
    # Logging Configuration
    LOG_RETENTION_DAYS = 30
    MAX_LOG_FILE_SIZE = 50 * 1024 * 1024  # 50MB


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output"""
    
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
    
    # Create formatters
    console_formatter = ColoredFormatter(
        '%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    
    file_formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, LOG_LEVEL))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, LOG_LEVEL))
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler - Main log
    today = datetime.now(IST).strftime('%Y-%m-%d')
    main_log_file = LOGS_DIR / f"trading_{today}.log"
    
    file_handler = RotatingFileHandler(
        main_log_file,
        maxBytes=TradingConfig.MAX_LOG_FILE_SIZE,
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # Trade-specific logger
    trade_logger = logging.getLogger('trades')
    trade_log_file = LOGS_DIR / f"trades_{today}.log"
    
    trade_handler = RotatingFileHandler(
        trade_log_file,
        maxBytes=TradingConfig.MAX_LOG_FILE_SIZE,
        backupCount=10,
        encoding='utf-8'
    )
    trade_handler.setLevel(logging.INFO)
    trade_handler.setFormatter(file_formatter)
    trade_logger.addHandler(trade_handler)
    trade_logger.setLevel(logging.INFO)
    
    return logging.getLogger(__name__), trade_logger


# Setup loggers
logger, trade_logger = setup_logging()


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
    stt: float
    exchange_charges: float
    gst: float
    sebi_charges: float
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
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging"""
        return asdict(self)


@dataclass
class MarketDepth:
    """5-level market depth structure"""
    
    timestamp: float
    bids: List[Tuple[float, int, int]]
    asks: List[Tuple[float, int, int]]
    
    @property
    def spread(self) -> float:
        if self.bids and self.asks:
            return self.asks[0][0] - self.bids[0][0]
        return float('inf')
    
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


class TradingMetrics:
    """Trading performance metrics calculator"""
    
    def __init__(self):
        self.trades: List[TradeReport] = []
        self.daily_pnl = 0
        self.total_brokerage = 0
        self.max_drawdown = 0
        self.peak_capital = 0
    
    def add_trade(self, trade: TradeReport):
        """Add completed trade to metrics"""
        self.trades.append(trade)
        self.daily_pnl += trade.net_pnl
        self.total_brokerage += (trade.brokerage + trade.stt + 
                                trade.exchange_charges + trade.gst + trade.sebi_charges)
    
    def calculate_metrics(self) -> Dict:
        """Calculate comprehensive trading metrics"""
        if not self.trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'avg_winner': 0,
                'avg_loser': 0,
                'largest_winner': 0,
                'largest_loser': 0,
                'net_pnl': 0,
                'gross_pnl': 0,
                'total_costs': 0,
                'roi': 0
            }
        
        winning_trades = [t for t in self.trades if t.net_pnl > 0]
        losing_trades = [t for t in self.trades if t.net_pnl <= 0]
        
        total_gross_profit = sum(t.gross_pnl for t in winning_trades)
        total_gross_loss = abs(sum(t.gross_pnl for t in losing_trades))
        
        metrics = {
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
        
        return metrics


class Position:
    """Enhanced Position class with comprehensive tracking"""
    
    def __init__(self, symbol: str, side: str, entry_price: float, quantity: int, 
                 entry_time: datetime, option_type: str, strike: float, 
                 expected_profit: float, deployed_capital: float):
        
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
        self.order_id = None
        self.stop_loss_order_id = None
        self.order_status = "PENDING"
        self.is_position_verified = False
        
        # Performance tracking
        self.pnl_history = []
        self.trailing_updates = []
        
        trade_logger.info(f"NEW POSITION CREATED: {self.trade_id} - {symbol} {side} {quantity}@{entry_price}")
    
    def calculate_all_charges(self, entry_price: float, exit_price: float, quantity: int) -> Dict[str, float]:
        """Calculate all trading charges comprehensively"""
        
        turnover = (entry_price + exit_price) * quantity
        
        # Brokerage (fixed per lot)
        num_lots = quantity / TradingConfig.LOT_SIZE
        brokerage = TradingConfig.BROKERAGE_PER_LOT * 2 * num_lots  # Entry + Exit
        
        # STT (Securities Transaction Tax) - only on sell side for options
        stt = exit_price * quantity * TradingConfig.STT_RATE
        
        # Exchange charges
        exchange_charges = turnover * TradingConfig.EXCHANGE_CHARGES_RATE
        
        # SEBI charges
        sebi_charges = turnover * TradingConfig.SEBI_CHARGES_RATE
        
        # GST on (brokerage + exchange charges + sebi charges)
        taxable_amount = brokerage + exchange_charges + sebi_charges
        gst = taxable_amount * TradingConfig.GST_RATE
        
        total_charges = brokerage + stt + exchange_charges + sebi_charges + gst
        
        return {
            'brokerage': round(brokerage, 2),
            'stt': round(stt, 2),
            'exchange_charges': round(exchange_charges, 2),
            'sebi_charges': round(sebi_charges, 2),
            'gst': round(gst, 2),
            'total_charges': round(total_charges, 2)
        }
    
    def update_stops(self, stop_loss: float, target: float):
        """Update stop loss and target with proper capital-based calculation"""
        if self.initial_stop is None:
            # Stop loss = 5% of capital deployed in THIS trade
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
        
        trade_logger.info(f"STOPS_UPDATED: {self.trade_id} - Stop: {self.stop_loss:.2f}, Target: {self.target:.2f}")
    
    def round_price_for_order(self, price: float, is_buy: bool) -> float:
        """Round price to 0.05 multiples for valid orders"""
        try:
            if not isinstance(price, (int, float)) or price <= 0:
                logger.error(f"Invalid price for rounding: {price}")
                return 0.05
            
            if is_buy:
                rounded_price = int(price / 0.05) * 0.05
            else:
                rounded_price = int((price + 0.049) / 0.05) * 0.05
            
            final_price = max(0.05, round(rounded_price, 2))
            return final_price
            
        except Exception as e:
            logger.error(f"Error rounding price {price}: {e}")
            return max(0.05, round(float(price), 2))
    
    def should_start_trailing(self, current_price: float) -> bool:
        """Check if trailing should start based on profit threshold"""
        if self.side != "BUY" or not self.trailing_start_price:
            return False
        
        current_profit = (current_price - self.entry_price) * self.quantity
        
        if current_profit >= TradingConfig.TRAILING_START_THRESHOLD:
            if not self.is_trailing_stop_active:
                logger.info(f"{Fore.GREEN}🎯 STARTING PROFIT-BASED TRAILING: {self.trade_id}")
                logger.info(f"   Profit ₹{current_profit:.0f} >= ₹{TradingConfig.TRAILING_START_THRESHOLD}{Style.RESET_ALL}")
                trade_logger.info(f"TRAILING_STARTED: {self.trade_id} - Profit: {current_profit:.0f}")
            return True
        
        return False
    
    def update_trailing_stop(self, current_price: float, kite_instance=None) -> bool:
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
                trade_logger.info(f"TARGET_REMOVED: {self.trade_id}")
            
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
                
                trade_logger.info(f"INITIAL_TRAILING: {self.trade_id} - Level: {trailing_level}, Stop: {target_stop_price:.2f}")
                
                return self._place_trailing_stop_order(target_stop_price, kite_instance)
            
            # Update to next level
            if trailing_level > self.current_trailing_level:
                old_level = self.current_trailing_level
                self.current_trailing_level = trailing_level
                self.trailing_stop_price = target_stop_price
                
                logger.info(f"{Fore.GREEN}🔄 TRAILING UPDATE: {self.trade_id}")
                logger.info(f"   Level: {old_level} → {trailing_level}")
                logger.info(f"   Stop: ₹{target_stop_price:.2f}{Style.RESET_ALL}")
                
                trade_logger.info(f"TRAILING_UPDATE: {self.trade_id} - Level: {old_level}->{trailing_level}, Stop: {target_stop_price:.2f}")
                
                return self._place_trailing_stop_order(target_stop_price, kite_instance)
            
            # Update max favorable price
            if current_price > self.max_favorable_price:
                self.max_favorable_price = current_price
            
            return False
            
        except Exception as e:
            logger.error(f"Error updating trailing stop for {self.trade_id}: {e}")
            return False
    
    def _place_trailing_stop_order(self, stop_price: float, kite_instance=None) -> bool:
        """Place trailing stop order with proper error handling"""
        try:
            if not kite_instance or not LIVE_TRADING_MODE:
                return True
            
            # Cancel existing stop order
            if self.stop_loss_order_id:
                try:
                    kite_instance.cancel_order(variety="regular", order_id=self.stop_loss_order_id)
                    logger.info(f"Cancelled old stop order: {self.stop_loss_order_id}")
                except Exception as cancel_error:
                    logger.warning(f"Could not cancel old stop: {cancel_error}")
            
            # Place new stop order
            rounded_stop_price = self.round_price_for_order(stop_price, is_buy=False)
            trigger_price = rounded_stop_price
            limit_price = rounded_stop_price - 0.05
            
            try:
                new_stop_order_id = kite_instance.place_order(
                    variety="regular",
                    exchange="NFO",
                    tradingsymbol=self.symbol,
                    transaction_type="SELL",
                    quantity=self.quantity,
                    product="MIS",
                    order_type="SL",
                    price=limit_price,
                    trigger_price=trigger_price,
                )
                
                self.stop_loss_order_id = new_stop_order_id
                self.stop_loss = rounded_stop_price
                self.original_stop_reinforced = False
                
                logger.info(f"{Fore.GREEN}✅ TRAILING STOP PLACED: {new_stop_order_id}{Style.RESET_ALL}")
                trade_logger.info(f"TRAILING_STOP_PLACED: {self.trade_id} - OrderID: {new_stop_order_id}, Price: {rounded_stop_price:.2f}")
                
                return True
                
            except Exception as order_error:
                logger.error(f"{Fore.RED}❌ TRAILING STOP FAILED: {order_error}{Style.RESET_ALL}")
                
                # Handle market price error
                error_msg = str(order_error).lower()
                if "market price" in error_msg and "less than" in error_msg:
                    return self._handle_stop_price_error(kite_instance)
                
                return False
                
        except Exception as e:
            logger.error(f"Error placing trailing stop for {self.trade_id}: {e}")
            return False
    
    def _handle_stop_price_error(self, kite_instance) -> bool:
        """Handle stop price error by reinforcing original stop"""
        try:
            logger.warning(f"{Fore.YELLOW}⚠️ MARKET PRICE ERROR: Reinforcing original stop for {self.trade_id}{Style.RESET_ALL}")
            
            original_trigger = self.round_price_for_order(self.original_stop, is_buy=False)
            original_limit = original_trigger - 0.05
            
            fallback_order_id = kite_instance.place_order(
                variety="regular",
                exchange="NFO",
                tradingsymbol=self.symbol,
                transaction_type="SELL",
                quantity=self.quantity,
                product="MIS",
                order_type="SL",
                price=original_limit,
                trigger_price=original_trigger,
            )
            
            self.stop_loss_order_id = fallback_order_id
            self.stop_loss = self.original_stop
            self.trailing_stop_price = self.original_stop
            self.original_stop_reinforced = True
            
            logger.info(f"{Fore.GREEN}✅ ORIGINAL STOP REINFORCED: {fallback_order_id}{Style.RESET_ALL}")
            trade_logger.info(f"ORIGINAL_STOP_REINFORCED: {self.trade_id} - OrderID: {fallback_order_id}")
            
            return True
            
        except Exception as fallback_error:
            logger.error(f"{Fore.RED}❌ FALLBACK STOP FAILED: {fallback_error}{Style.RESET_ALL}")
            trade_logger.error(f"FALLBACK_STOP_FAILED: {self.trade_id} - Error: {fallback_error}")
            return False
    
    def check_emergency_exit(self, current_price: float) -> bool:
        """Check if emergency exit is needed"""
        if self.side == "BUY" and current_price <= self.original_stop:
            logger.error(f"{Fore.RED}🚨 EMERGENCY EXIT NEEDED: {self.trade_id}{Style.RESET_ALL}")
            trade_logger.error(f"EMERGENCY_EXIT: {self.trade_id} - Price: {current_price:.2f}, Original Stop: {self.original_stop:.2f}")
            return True
        return False
    
    def close(self, exit_price: float, exit_time: datetime, exit_reason: str = "UNKNOWN") -> TradeReport:
        """Close position and generate comprehensive trade report"""
        
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.exit_reason = exit_reason
        
        # Calculate gross P&L
        if self.side == "BUY":
            gross_pnl = (exit_price - self.entry_price) * self.quantity
        else:
            gross_pnl = (self.entry_price - exit_price) * self.quantity
        
        # Calculate all charges
        charges = self.calculate_all_charges(self.entry_price, exit_price, self.quantity)
        
        # Calculate net P&L
        net_pnl = gross_pnl - charges['total_charges']
        
        # Calculate holding duration
        holding_duration = (exit_time - self.entry_time).total_seconds() / 60  # minutes
        
        # Calculate ROI
        roi_percentage = (net_pnl / self.deployed_capital) * 100 if self.deployed_capital > 0 else 0
        
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
            brokerage=charges['brokerage'],
            stt=charges['stt'],
            exchange_charges=charges['exchange_charges'],
            gst=charges['gst'],
            sebi_charges=charges['sebi_charges'],
            net_pnl=net_pnl,
            deployed_capital=self.deployed_capital,
            max_favorable_price=self.max_favorable_price,
            trailing_activated=self.is_trailing_stop_active,
            target_hit=(exit_reason == "TARGET_ACHIEVED"),
            stop_loss_hit=(exit_reason in ["STOP_LOSS_HIT", "EMERGENCY_EXIT"]),
            exit_reason=exit_reason,
            holding_duration_minutes=holding_duration,
            roi_percentage=roi_percentage
        )
        
        # Log trade completion
        logger.info(f"{Fore.CYAN}TRADE COMPLETED: {self.trade_id}")
        logger.info(f"  Gross P&L: ₹{gross_pnl:.2f}")
        logger.info(f"  Total Charges: ₹{charges['total_charges']:.2f}")
        logger.info(f"  Net P&L: ₹{net_pnl:.2f}")
        logger.info(f"  ROI: {roi_percentage:.2f}%")
        logger.info(f"  Duration: {holding_duration:.1f} minutes{Style.RESET_ALL}")
        
        # Detailed trade log
        trade_logger.info(f"TRADE_COMPLETED: {json.dumps(trade_report.to_dict(), default=str, indent=2)}")
        
        return trade_report


class ImprovedProbabilityEngine:
    """Probability engine for options analysis"""
    
    def __init__(self, spot: float, strike: float, tte: float, rate: float, iv: float):
        self.S = spot
        self.K = strike
        self.T = max(tte, 0.001)
        self.r = rate
        self.σ = iv
        self.skew = self._calculate_skew()
        self._update_greeks()
    
    def _calculate_skew(self) -> float:
        """Calculate implied skew based on moneyness"""
        moneyness = self.S / self.K
        if moneyness > 1.02:
            return -0.3
        elif moneyness < 0.98:
            return 0.3
        else:
            return 0
    
    def _update_greeks(self):
        """Calculate Greeks with skew adjustment"""
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.σ**2) * self.T) / (self.σ * np.sqrt(self.T))
        d2 = d1 - self.σ * np.sqrt(self.T)
        
        skew_adjustment = self.skew * np.sqrt(self.T)
        d1_adj = d1 + skew_adjustment
        d2_adj = d2 + skew_adjustment
        
        self.delta = norm.cdf(d1_adj)
        self.gamma = norm.pdf(d1_adj) / (self.S * self.σ * np.sqrt(self.T))
        self.vega = self.S * norm.pdf(d1_adj) * np.sqrt(self.T) / 100
        self.theta = -(self.S * norm.pdf(d1_adj) * self.σ) / (2 * np.sqrt(self.T)) / 365
    
    def get_confidence_zones(self, confidence_levels=[0.68, 0.95]) -> Dict:
        """Get confidence zones for price prediction"""
        try:
            center_price = self.S
            daily_volatility = self.σ / np.sqrt(252)
            time_adjusted_vol = daily_volatility * np.sqrt(max(self.T * 252, 1))
            
            zones = {}
            for conf in confidence_levels:
                z_score = norm.ppf((1 + conf) / 2)
                price_range = center_price * time_adjusted_vol * z_score
                min_range = center_price * 0.005
                price_range = max(price_range, min_range)
                
                if self.T < 0.1:
                    max_range = center_price * 0.05
                    price_range = min(price_range, max_range)
                
                lower_bound = max(center_price - price_range, center_price * 0.8)
                upper_bound = min(center_price + price_range, center_price * 1.2)
                
                zones[conf] = {
                    'lower': lower_bound,
                    'upper': upper_bound,
                    'peak': center_price,
                    'width': upper_bound - lower_bound,
                }
            
            return zones
            
        except Exception as e:
            logger.error(f"Error calculating confidence zones: {e}")
            return {
                0.68: {'lower': self.S * 0.98, 'upper': self.S * 1.02, 'peak': self.S, 'width': self.S * 0.04},
                0.95: {'lower': self.S * 0.95, 'upper': self.S * 1.05, 'peak': self.S, 'width': self.S * 0.10},
            }


class TrendAnalyzer:
    """Trend analysis for market direction"""
    
    def __init__(self, lookback_periods=50):
        self.price_history = deque(maxlen=lookback_periods)
        self.volume_history = deque(maxlen=lookback_periods)
    
    def update(self, price: float, volume: int):
        """Update price and volume history"""
        self.price_history.append(price)
        self.volume_history.append(volume)
    
    def get_trend_signal(self) -> Dict:
        """Get trend signal for trading"""
        if len(self.price_history) < 20:
            return {'trend': 'NEUTRAL', 'strength': 0, 'vwap': 0}
        
        prices = np.array(self.price_history)
        volumes = np.array(self.volume_history)
        
        # VWAP calculation
        vwap = np.sum(prices * volumes) / np.sum(volumes) if np.sum(volumes) > 0 else prices[-1]
        
        # Trend calculation
        x = np.arange(len(prices))
        slope = np.polyfit(x, prices, 1)[0]
        price_std = np.std(prices)
        strength = abs(slope) / price_std if price_std > 0 else 0
        
        # Determine trend direction
        if slope > price_std * 0.01:
            trend = 'BULLISH'
        elif slope < -price_std * 0.01:
            trend = 'BEARISH'
        else:
            trend = 'NEUTRAL'
        
        # Volume analysis
        recent_volume = np.mean(volumes[-5:])
        avg_volume = np.mean(volumes[:-5]) if len(volumes) > 5 else recent_volume
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
        
        # Adjust strength based on volume
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


def is_market_hours() -> bool:
    """Check if current time is within market hours (IST)"""
    now = datetime.now(IST)
    market_start = now.replace(
        hour=TradingConfig.MARKET_START_HOUR, 
        minute=TradingConfig.MARKET_START_MINUTE, 
        second=0, 
        microsecond=0
    )
    market_end = now.replace(
        hour=TradingConfig.MARKET_END_HOUR, 
        minute=TradingConfig.MARKET_END_MINUTE, 
        second=0, 
        microsecond=0
    )
    
    return market_start <= now <= market_end


def is_trading_day() -> bool:
    """Check if today is a trading day (Monday-Friday, excluding holidays)"""
    now = datetime.now(IST)
    # Monday = 0, Sunday = 6
    return now.weekday() < 5  # Monday to Friday


class NIFTYIntradayEngine:
    """Main NIFTY Options Intraday Trading Engine"""
    
    def __init__(self, capital: float):
        self.capital = capital
        self.available_capital = capital
        self.deployed_capital = 0
        
        # Initialize Kite connection
        self.kite = KiteConnect(api_key=API_KEY) if API_KEY else None
        if self.kite and ACCESS_TOKEN:
            self.kite.set_access_token(ACCESS_TOKEN)
        
        # Trading state
        self.positions: Dict[str, Position] = {}
        self.trading_metrics = TradingMetrics()
        self.probability_engines: Dict[str, ImprovedProbabilityEngine] = {}
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
        
        # Position tracking
        self.last_position_sync = datetime.min
        self.live_positions = {}
        self.pending_orders = {}
        self.margin_used = 0
        
        # Performance tracking
        self.trades_today = 0
        self.last_trade_time = {}
        self.last_signal_time = {}
        
        logger.info(f"{Fore.GREEN}NIFTY Engine Initialized{Style.RESET_ALL}")
        logger.info(f"Capital: ₹{capital:,.2f}")
        logger.info(f"Mode: {'LIVE TRADING' if LIVE_TRADING_MODE else 'SIMULATION'}")
        logger.info(f"Trailing: Start ₹{TradingConfig.TRAILING_START_THRESHOLD}, Increment ₹{TradingConfig.TRAILING_STOP_INCREMENT}")
        
        if LIVE_TRADING_MODE:
            print(f"\n{Fore.RED}{'='*80}{Style.RESET_ALL}")
            print(f"{Fore.RED}⚠️  LIVE TRADING MODE ENABLED ⚠️{Style.RESET_ALL}")
            print(f"{Fore.RED}REAL MONEY WILL BE USED!{Style.RESET_ALL}")
            print(f"{Fore.RED}{'='*80}{Style.RESET_ALL}\n")
        else:
            print(f"\n{Fore.YELLOW}{'='*80}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}🔧 SIMULATION MODE 🔧{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}{'='*80}{Style.RESET_ALL}\n")
    
    async def check_budget_and_margin(self, required_capital: float) -> bool:
        """Check if sufficient capital is available"""
        try:
            if not LIVE_TRADING_MODE:
                return True
            
            if not self.kite:
                logger.error("Kite connection not available")
                return False
            
            margins = self.kite.margins()
            available_cash = margins.get('equity', {}).get('available', {}).get('cash', 0)
            
            pending_allocation = sum(self.pending_orders.values())
            capital_utilization = (self.margin_used + pending_allocation + required_capital) / self.capital
            
            if capital_utilization > TradingConfig.MAX_CAPITAL_UTILIZATION:
                logger.warning(f"{Fore.YELLOW}CAPITAL LIMIT: Would use {capital_utilization:.1%}{Style.RESET_ALL}")
                return False
            
            effective_available = available_cash - pending_allocation
            if effective_available < required_capital:
                logger.warning(f"{Fore.YELLOW}INSUFFICIENT CASH: Need ₹{required_capital:,.2f}, Have ₹{effective_available:,.2f}{Style.RESET_ALL}")
                return False
            
            logger.info(f"{Fore.GREEN}BUDGET CHECK PASSED: Using {capital_utilization:.1%} of capital{Style.RESET_ALL}")
            return True
            
        except Exception as e:
            logger.error(f"Error checking budget: {e}")
            return False
    
    def is_in_focus_mode(self) -> bool:
        """Check if in focus mode (high capital utilization)"""
        capital_utilization = self.margin_used / self.capital if self.capital > 0 else 0
        return capital_utilization >= TradingConfig.FOCUS_MODE_THRESHOLD
    
    def calculate_position_size_enhanced(self, confidence: float, entry_price: float) -> Tuple[int, float]:
        """Calculate position size based on confidence and available capital"""
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
        """Calculate expected profit with all charges"""
        
        # Create temporary position for charge calculation
        temp_charges_target = Position.calculate_all_charges(
            None, entry_price, target_price, quantity
        )
        temp_charges_stop = Position.calculate_all_charges(
            None, entry_price, stop_price, quantity
        )
        
        gross_profit = abs(target_price - entry_price) * quantity
        gross_loss = abs(entry_price - stop_price) * quantity
        
        net_profit = gross_profit - temp_charges_target['total_charges']
        net_loss = gross_loss + temp_charges_stop['total_charges']
        
        risk_reward = net_profit / net_loss if net_loss > 0 else 0
        
        is_worth_trading = (
            net_profit >= TradingConfig.MIN_PROFIT_AFTER_BROKERAGE and
            risk_reward >= TradingConfig.MIN_RISK_REWARD_RATIO
        )
        
        return {
            'net_profit': net_profit,
            'net_loss': net_loss,
            'risk_reward_ratio': risk_reward,
            'total_charges_profit': temp_charges_target['total_charges'],
            'total_charges_loss': temp_charges_stop['total_charges'],
            'is_worth_trading': is_worth_trading,
        }
    
    async def verify_position_created(self, symbol: str, expected_quantity: int, timeout_seconds: int = 30) -> bool:
        """Verify position creation after order placement"""
        try:
            if not LIVE_TRADING_MODE or not self.kite:
                return True
            
            start_time = time.time()
            while time.time() - start_time < timeout_seconds:
                try:
                    positions = self.kite.positions()
                    
                    for pos in positions.get('net', []):
                        if (pos['tradingsymbol'] == symbol and 
                            abs(pos['quantity']) == expected_quantity):
                            logger.info(f"{Fore.GREEN}✅ POSITION VERIFIED: {symbol} Qty={pos['quantity']}{Style.RESET_ALL}")
                            return True
                    
                    await asyncio.sleep(2)
                    
                except Exception as pos_error:
                    logger.warning(f"Error checking positions: {pos_error}")
                    await asyncio.sleep(2)
            
            logger.error(f"{Fore.RED}❌ POSITION NOT VERIFIED: {symbol} within {timeout_seconds}s{Style.RESET_ALL}")
            return False
            
        except Exception as e:
            logger.error(f"Error verifying position: {e}")
            return False
    
    async def sync_positions_with_broker(self):
        """Sync positions with broker"""
        try:
            if not LIVE_TRADING_MODE or not self.kite:
                return
            
            current_time = datetime.now()
            if current_time - self.last_position_sync < timedelta(minutes=1):
                return
            
            self.last_position_sync = current_time
            
            broker_positions = self.kite.positions()
            nifty_positions = []
            
            for pos in broker_positions.get('net', []):
                if pos['tradingsymbol'].startswith('NIFTY') and pos['quantity'] != 0:
                    nifty_positions.append(pos)
            
            self.live_positions = {pos['tradingsymbol']: pos for pos in nifty_positions}
            
            self.deployed_capital = sum(
                abs(pos.get('quantity', 0)) * pos.get('last_price', 0)
                for pos in nifty_positions
            )
            
            self.margin_used = sum(
                abs(pos.get('quantity', 0)) * pos.get('average_price', 0)
                for pos in nifty_positions
            )
            
        except Exception as e:
            logger.error(f"Error syncing positions: {e}")
    
    def get_tradeable_strikes(self) -> List[Dict]:
        """Get tradeable option strikes"""
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
        """Execute trade with comprehensive tracking"""
        try:
            # Create temporary order ID for tracking
            temp_order_id = f"temp_{int(time.time())}"
            self.pending_orders[temp_order_id] = deployed_capital
            
            # Create position object
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
            )
            
            # Setup stops
            position.update_stops(stop_price, target_price)
            
            logger.info(f"{Fore.GREEN}EXECUTING TRADE:{Style.RESET_ALL}")
            logger.info(f"Symbol: {symbol}")
            logger.info(f"Entry: ₹{price:.2f} | Target: ₹{target_price:.2f} | Stop: ₹{stop_price:.2f}")
            logger.info(f"Quantity: {quantity} | Capital: ₹{deployed_capital:,.2f}")
            
            if LIVE_TRADING_MODE and self.kite:
                try:
                    # Place entry order
                    actual_order_id = self.kite.place_order(
                        variety="regular",
                        exchange="NFO",
                        tradingsymbol=symbol,
                        transaction_type=side,
                        quantity=quantity,
                        product="MIS",
                        order_type="LIMIT",
                        price=price,
                    )
                    
                    logger.info(f"{Fore.GREEN}✅ ENTRY ORDER PLACED: {actual_order_id}{Style.RESET_ALL}")
                    position.order_id = actual_order_id
                    
                    # Wait for order execution
                    await asyncio.sleep(2)
                    
                    # Verify position creation
                    position_verified = await self.verify_position_created(symbol, quantity, timeout_seconds=10)
                    
                    if position_verified:
                        position.is_position_verified = True
                        position.order_status = "EXECUTED"
                        
                        # Place initial stop loss
                        logger.info(f"{Fore.YELLOW}PLACING INITIAL STOP LOSS...{Style.RESET_ALL}")
                        
                        trigger_price = position.round_price_for_order(stop_price, is_buy=False)
                        limit_price = trigger_price - 0.05
                        
                        try:
                            stop_order_id = self.kite.place_order(
                                variety="regular",
                                exchange="NFO",
                                tradingsymbol=symbol,
                                transaction_type="SELL",
                                quantity=quantity,
                                product="MIS",
                                order_type="SL",
                                price=limit_price,
                                trigger_price=trigger_price,
                            )
                            
                            position.stop_loss_order_id = stop_order_id
                            logger.info(f"{Fore.GREEN}✅ INITIAL STOP LOSS PLACED: {stop_order_id}{Style.RESET_ALL}")
                            
                        except Exception as stop_error:
                            logger.error(f"{Fore.RED}❌ STOP LOSS FAILED: {stop_error}{Style.RESET_ALL}")
                    
                    else:
                        logger.error(f"{Fore.RED}❌ ENTRY ORDER FAILED - Position not verified{Style.RESET_ALL}")
                        position.order_status = "REJECTED"
                        del self.pending_orders[temp_order_id]
                        return
                
                except Exception as order_error:
                    logger.error(f"{Fore.RED}❌ ORDER PLACEMENT FAILED: {order_error}{Style.RESET_ALL}")
                    del self.pending_orders[temp_order_id]
                    return
            
            else:
                # Simulation mode
                logger.info(f"{Fore.YELLOW}[SIMULATED] Trade executed{Style.RESET_ALL}")
                position.is_position_verified = True
                position.order_status = "EXECUTED"
            
            # Add to active positions
            self.positions[token] = position
            self.last_trade_time[token] = datetime.now()
            self.trades_today += 1
            
            # Update capital tracking
            self.deployed_capital += deployed_capital
            self.margin_used += deployed_capital
            del self.pending_orders[temp_order_id]
            
            logger.info(f"{Fore.GREEN}✅ TRADE SETUP COMPLETE: {symbol}{Style.RESET_ALL}")
            
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            if temp_order_id in self.pending_orders:
                del self.pending_orders[temp_order_id]
    
    async def close_position(self, token: int, current_price: float, exit_reason: str = "UNKNOWN"):
        """Close position and generate report"""
        try:
            position = self.positions[token]
            
            logger.info(f"{Fore.YELLOW}CLOSING POSITION: {position.trade_id} - Reason: {exit_reason}{Style.RESET_ALL}")
            
            if LIVE_TRADING_MODE and self.kite:
                exit_side = "SELL" if position.side == "BUY" else "BUY"
                exit_price_rounded = self.round_price_for_order(current_price, is_buy=(exit_side == "BUY"))
                
                try:
                    # Cancel stop loss order if exists
                    if position.stop_loss_order_id:
                        try:
                            self.kite.cancel_order(variety="regular", order_id=position.stop_loss_order_id)
                            logger.info(f"Cancelled stop loss order: {position.stop_loss_order_id}")
                        except Exception as cancel_error:
                            logger.warning(f"Could not cancel stop loss: {cancel_error}")
                    
                    # Place exit order
                    exit_order_id = self.kite.place_order(
                        variety="regular",
                        exchange="NFO",
                        tradingsymbol=position.symbol,
                        transaction_type=exit_side,
                        quantity=position.quantity,
                        product="MIS",
                        order_type="LIMIT",
                        price=exit_price_rounded,
                    )
                    
                    logger.info(f"{Fore.GREEN}✅ EXIT ORDER PLACED: {exit_order_id}{Style.RESET_ALL}")
                    
                except Exception as exit_error:
                    logger.error(f"{Fore.RED}❌ EXIT ORDER FAILED: {exit_error}{Style.RESET_ALL}")
            
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
                    position.update_trailing_stop(current_price, self.kite)
                    
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
            
            if (token not in self.probability_engines or 
                token not in self.monitored_options or 
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
            await self.sync_positions_with_broker()
            
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
                    
                    if position_size > 0 and await self.check_budget_and_margin(required_capital):
                        
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
                        logger.info(f"  Net Profit Potential: ₹{trade_analysis['net_profit']:.2f}")
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
                                expected_profit=trade_analysis['net_profit'],
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
                
                # Setup probability engine
                days_to_expiry = max(1, (opt['expiry_dt'].date() - datetime.now().date()).days)
                tte_years = days_to_expiry / 365
                
                moneyness = abs(opt['strike'] - self.spot_price) / self.spot_price
                base_iv = 0.15
                iv_adjustment = moneyness * 0.3
                estimated_iv = base_iv + iv_adjustment
                
                self.probability_engines[opt['instrument_token']] = ImprovedProbabilityEngine(
                    spot=self.spot_price,
                    strike=opt['strike'],
                    tte=tte_years,
                    rate=0.065,
                    iv=estimated_iv,
                )
            
            tokens = [opt['instrument_token'] for opt in selected_options]
            
            logger.info(f"Monitoring {len(tokens)} options for profit-based trailing")
            
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
                logger.info(f"{Fore.GREEN}Connected to market data stream{Style.RESET_ALL}")
            
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
                if tick_count % 50 == 0:
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
        if current_time - self.last_display_time < 5:
            return
        
        self.last_display_time = current_time
        
        focus_mode = self.is_in_focus_mode()
        capital_utilization = self.margin_used / self.capital if self.capital > 0 else 0
        
        print(f"\n{Fore.YELLOW}{'='*80}{Style.RESET_ALL}")
        if focus_mode:
            print(f"{Fore.RED}🎯 FOCUS MODE - Managing Existing Trades Only ({capital_utilization:.1%} capital used){Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}PROFIT-BASED TRAILING OPTIONS - {datetime.now(IST).strftime('%H:%M:%S')}{Style.RESET_ALL}")
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
                if position.original_stop_reinforced:
                    status_parts.append("⚠️ REINF")
                
                position_status = " ".join(status_parts)
            else:
                position_status = "⚪" if not focus_mode else "🎯 FOCUS"
            
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
        available_capital = self.capital * TradingConfig.MAX_CAPITAL_UTILIZATION - self.margin_used
        metrics = self.trading_metrics.calculate_metrics()
        
        print(f"{Fore.CYAN}Capital: ₹{self.capital + metrics['net_pnl']:,.2f} | P&L: ₹{metrics['net_pnl']:,.2f} | Win Rate: {metrics['win_rate']:.1f}%{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Used: ₹{self.margin_used:,.2f} ({capital_utilization:.1%}) | Available: ₹{available_capital:,.2f} | Focus: {'YES' if focus_mode else 'NO'}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Positions: {len(self.positions)} | Trades: {metrics['total_trades']} | Rejected: 0{Style.RESET_ALL}")
    
    async def generate_daily_report(self):
        """Generate comprehensive daily report"""
        try:
            metrics = self.trading_metrics.calculate_metrics()
            today = datetime.now(IST).strftime('%Y-%m-%d')
            
            # Create report content
            report_content = f"""
# NIFTY Options Trading Daily Report
Date: {today}
Generated: {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S IST')}

## Trading Summary
- **Total Trades**: {metrics['total_trades']}
- **Winning Trades**: {metrics['winning_trades']}
- **Losing Trades**: {metrics['losing_trades']}
- **Win Rate**: {metrics['win_rate']:.2f}%
- **Profit Factor**: {metrics['profit_factor']:.2f}

## Financial Performance
- **Net P&L**: ₹{metrics['net_pnl']:,.2f}
- **Gross P&L**: ₹{metrics['gross_pnl']:,.2f}
- **Total Costs**: ₹{metrics['total_costs']:,.2f}
- **ROI**: {metrics['roi']:.2f}%
- **Average Winner**: ₹{metrics['avg_winner']:,.2f}
- **Average Loser**: ₹{metrics['avg_loser']:,.2f}
- **Largest Winner**: ₹{metrics['largest_winner']:,.2f}
- **Largest Loser**: ₹{metrics['largest_loser']:,.2f}

## Trading Configuration
- **Capital**: ₹{self.capital:,.2f}
- **Mode**: {'LIVE TRADING' if LIVE_TRADING_MODE else 'SIMULATION'}
- **Trailing Start**: ₹{TradingConfig.TRAILING_START_THRESHOLD}
- **Trailing Increment**: ₹{TradingConfig.TRAILING_STOP_INCREMENT}
- **Max Capital Utilization**: {TradingConfig.MAX_CAPITAL_UTILIZATION:.1%}
- **Stop Loss**: {TradingConfig.STOP_LOSS_PERCENTAGE:.1%} of deployed capital

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
- **Costs**: ₹{trade.brokerage + trade.stt + trade.exchange_charges + trade.gst + trade.sebi_charges:.2f}
- **Net P&L**: ₹{trade.net_pnl:.2f}
- **ROI**: {trade.roi_percentage:.2f}%
- **Duration**: {trade.holding_duration_minutes:.1f} minutes
- **Exit Reason**: {trade.exit_reason}
- **Trailing Used**: {'Yes' if trade.trailing_activated else 'No'}
"""
            
            # Save report
            report_file = REPORTS_DIR / f"daily_report_{today}.md"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            logger.info(f"{Fore.GREEN}Daily report generated: {report_file}{Style.RESET_ALL}")
            
            # Also save as JSON for programmatic access
            json_report = {
                'date': today,
                'generated': datetime.now(IST).isoformat(),
                'metrics': metrics,
                'trades': [trade.to_dict() for trade in self.trading_metrics.trades],
                'config': {
                    'capital': self.capital,
                    'live_mode': LIVE_TRADING_MODE,
                    'trailing_start': TradingConfig.TRAILING_START_THRESHOLD,
                    'trailing_increment': TradingConfig.TRAILING_STOP_INCREMENT,
                }
            }
            
            json_file = REPORTS_DIR / f"daily_report_{today}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(json_report, f, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"Error generating daily report: {e}")
    
    async def performance_monitor(self):
        """Monitor performance and generate periodic reports"""
        while self.is_running:
            try:
                await asyncio.sleep(300)  # 5 minutes
                await self.sync_positions_with_broker()
                
                metrics = self.trading_metrics.calculate_metrics()
                current_capital = self.capital + metrics['net_pnl']
                returns = metrics['roi']
                
                print(f"\n{Fore.CYAN}=== PERFORMANCE UPDATE ==={Style.RESET_ALL}")
                print(f"Capital: ₹{current_capital:,.2f} ({returns:+.2f}%)")
                print(f"P&L: ₹{metrics['net_pnl']:,.2f}")
                print(f"Positions: {len(self.positions)}")
                print(f"Win Rate: {metrics['win_rate']:.1f}%")
                print(f"Profit Factor: {metrics['profit_factor']:.2f}")
                
            except Exception as e:
                logger.error(f"Performance monitor error: {e}")
    
    async def start(self):
        """Start the trading engine"""
        try:
            if not is_market_hours():
                logger.warning("Market is closed. Engine will wait for market hours.")
                return
            
            self.is_running = True
            logger.info("Starting NIFTY Options Trading Engine...")
            
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
        print(f"\n{Fore.YELLOW}=== FINAL TRADING SUMMARY ==={Style.RESET_ALL}")
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Win Rate: {metrics['win_rate']:.1f}%")
        print(f"Net P&L: ₹{metrics['net_pnl']:,.2f}")
        print(f"ROI: {metrics['roi']:.2f}%")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        
        logger.info("Trading engine shutdown complete")


def signal_handler(signum, frame):
    """Handle termination signals"""
    logger.info("Received termination signal, shutting down...")
    sys.exit(0)


def schedule_trading():
    """Schedule trading for market hours"""
    
    def start_trading_job():
        """Job to start trading"""
        if is_trading_day() and is_market_hours():
            logger.info("Starting scheduled trading session...")
            asyncio.run(main_trading_loop())
        else:
            logger.info("Not a trading day or outside market hours")
    
    # Schedule trading start at 9:15 AM IST
    schedule.every().monday.at("09:15").do(start_trading_job)
    schedule.every().tuesday.at("09:15").do(start_trading_job)
    schedule.every().wednesday.at("09:15").do(start_trading_job)
    schedule.every().thursday.at("09:15").do(start_trading_job)
    schedule.every().friday.at("09:15").do(start_trading_job)
    
    logger.info("Trading scheduled for weekdays at 9:15 AM IST")
    
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute


async def main_trading_loop():
    """Main trading loop"""
    try:
        # Validate environment
        if not API_KEY or not ACCESS_TOKEN:
            logger.error("Missing required environment variables: KITE_API_KEY or KITE_ACCESS_TOKEN")
            print("\nRequired Environment Variables:")
            print("- KITE_API_KEY: Your Kite Connect API key")
            print("- KITE_ACCESS_TOKEN: Your Kite Connect access token")
            print("- TRADING_CAPITAL: Trading capital amount (default: 100000)")
            print("- LIVE_TRADING_MODE: true/false (default: false)")
            print("- LOG_LEVEL: DEBUG/INFO/WARNING/ERROR (default: INFO)")
            return
        
        # Display startup banner
        print(f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════╗
║         NIFTY OPTIONS TRADING SYSTEM v2.0               ║
║          Profit-Based Trailing Stops                    ║
║          ₹{TradingConfig.TRAILING_START_THRESHOLD} Start | ₹{TradingConfig.TRAILING_STOP_INCREMENT} Increments                     ║
╚══════════════════════════════════════════════════════════╝{Style.RESET_ALL}
        """)
        
        logger.info(f"Trading Capital: ₹{TRADING_CAPITAL:,.2f}")
        logger.info(f"Mode: {'LIVE TRADING' if LIVE_TRADING_MODE else 'SIMULATION'}")
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
{Fore.GREEN}NIFTY Options Trading System{Style.RESET_ALL}
{'='*50}

Environment Configuration:
- API Key: {'✓ Set' if API_KEY else '✗ Missing'}
- Access Token: {'✓ Set' if ACCESS_TOKEN else '✗ Missing'}
- Trading Capital: ₹{TRADING_CAPITAL:,.2f}
- Live Trading: {'✓ Enabled' if LIVE_TRADING_MODE else '✗ Disabled (Simulation)'}
- Log Level: {LOG_LEVEL}

Market Hours: {TradingConfig.MARKET_START_HOUR}:{TradingConfig.MARKET_START_MINUTE:02d} - {TradingConfig.MARKET_END_HOUR}:{TradingConfig.MARKET_END_MINUTE:02d} IST
    """)
    
    try:
        if len(sys.argv) > 1 and sys.argv[1] == '--schedule':
            # Run in scheduled mode
            logger.info("Starting in scheduled mode...")
            schedule_trading()
        else:
            # Run immediately if market hours
            if is_trading_day() and is_market_hours():
                logger.info("Market is open, starting trading immediately...")
                asyncio.run(main_trading_loop())
            else:
                logger.info("Market is closed or not a trading day")
                print(f"\nCurrent time: {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S IST')}")
                print("Use --schedule flag to run in scheduled mode")
                
    except KeyboardInterrupt:
        logger.info("Program interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
