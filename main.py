#!/usr/bin/env python3
"""
NIFTY Options Intraday Engine — Simulation Mode

Consumes live options tick data over the Kite Connect WebSocket and runs the full
intraday order lifecycle (entry, risk-budgeted stop, profit-based trailing, exit)
without placing real orders. Fills are modelled against the order book with
slippage so simulated P&L stays closer to live-exchange behaviour.
"""

from __future__ import annotations

import os
import sys
import time
import json
import queue
import signal
import logging
import threading
import asyncio
from collections import deque
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

try:
    from kiteconnect import KiteConnect, KiteTicker
    HAVE_KITE = True
except Exception:
    KiteConnect = KiteTicker = None
    HAVE_KITE = False

try:
    import pytz
    IST = pytz.timezone("Asia/Kolkata")
except Exception:
    IST = None


def now_ist() -> datetime:
    """Single source of truth for time — always IST-aware when pytz is present."""
    return datetime.now(IST) if IST else datetime.now()


class Config:
                           
    LOT_SIZE = 75
    BROKERAGE_PER_LOT_PER_SIDE = 40

                               
    MAX_OPEN_POSITIONS = 2
    MAX_DAILY_TRADES = int(os.getenv("MAX_DAILY_TRADES", "10"))
    MAX_CAPITAL_UTILIZATION = 0.80
    FOCUS_MODE_THRESHOLD = 0.70

                                                                           
    MAX_LOSS_PER_TRADE = float(os.getenv("MAX_LOSS_PER_TRADE", "600"))
    STOP_SANITY_FLOOR_PCT = 0.30
    TARGET_MULTIPLIER = 1.06
    MIN_RISK_REWARD = 0.75
    MIN_NET_PROFIT_PER_LOT = 100

                               
    MAX_DAILY_LOSS = float(os.getenv("MAX_DAILY_LOSS", "2500"))

                           
    TRAILING_START_PROFIT = 150
    TRAILING_STEP = 100

                   
    MIN_CONFIDENCE = 0.65
    SIGNAL_COOLDOWN_S = 10
    TRADE_COOLDOWN_S = 60
    SIGNAL_EVERY_N_TICKS = 5
    MAX_SPREAD_PCT = 0.02
    MIN_DEPTH_VOLUME = 100

                                  
    STOP_SLIPPAGE_TICKS = 2
    TICK_SIZE = 0.05

                        
    MKT_OPEN = (9, 15)
    MKT_CLOSE = (15, 30)

                                     
    RISK_FREE_RATE = 0.065


def setup_logging() -> logging.Logger:
    logs = Path("logs")
    logs.mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    fh = logging.FileHandler(logs / f"trading_{now_ist():%Y-%m-%d}.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s"))
    root.addHandler(fh)
    return logging.getLogger("engine")


log = setup_logging()


class BlackScholes:
    @staticmethod
    def d1_d2(S, K, T, r, sigma):
        if min(S, K, T, sigma) <= 0:
            return 0.0, 0.0
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return d1, d1 - sigma * np.sqrt(T)

    @staticmethod
    def price(S, K, T, r, sigma, kind="CE"):
        if T <= 0:
            return max(S - K, 0) if kind in ("CE", "CALL") else max(K - S, 0)
        d1, d2 = BlackScholes.d1_d2(S, K, T, r, sigma)
        if kind in ("CE", "CALL"):
            return max(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2), 0)
        return max(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1), 0)

    @staticmethod
    def greeks(S, K, T, r, sigma, kind="CE"):
        if min(S, K, T, sigma) <= 0:
            return {"delta": 0, "gamma": 0, "theta": 0, "vega": 0}
        d1, d2 = BlackScholes.d1_d2(S, K, T, r, sigma)
        delta = norm.cdf(d1) if kind in ("CE", "CALL") else norm.cdf(d1) - 1
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        base = (-S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        if kind in ("CE", "CALL"):
            theta = (base - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        else:
            theta = (base + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100
        return {"delta": round(delta, 4), "gamma": round(gamma, 6),
                "theta": round(theta, 3), "vega": round(vega, 3)}

    @staticmethod
    def implied_vol(S, K, T, r, mkt_price, kind="CE", iters=100):
        if mkt_price <= 0 or T <= 0:
            return 0.20
        sigma = 0.20
        for _ in range(iters):
            theo = BlackScholes.price(S, K, T, r, sigma, kind)
            diff = theo - mkt_price
            if abs(diff) < 1e-4:
                return round(sigma, 4)
            d1, _ = BlackScholes.d1_d2(S, K, T, r, sigma)
            vega = S * norm.pdf(d1) * np.sqrt(T)
            if vega == 0:
                break
            sigma = max(0.01, min(sigma - diff / vega, 5.0))
        return round(sigma, 4)


@dataclass
class MarketDepth:
    timestamp: float
    bids: List[Tuple[float, int, int]]
    asks: List[Tuple[float, int, int]]

    @property
    def best_bid(self) -> float:
        return self.bids[0][0] if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return self.asks[0][0] if self.asks else 0.0

    @property
    def spread(self) -> float:
        return (self.best_ask - self.best_bid) if (self.bids and self.asks) else float("inf")

    @property
    def mid(self) -> float:
        return (self.best_bid + self.best_ask) / 2 if (self.bids and self.asks) else 0.0

    @property
    def volume_imbalance(self) -> float:
        b = sum(x[1] for x in self.bids[:5])
        a = sum(x[1] for x in self.asks[:5])
        return (b - a) / (b + a) if (b + a) else 0.0

    @property
    def total_depth_volume(self) -> int:
        return sum(x[1] for x in self.bids[:5]) + sum(x[1] for x in self.asks[:5])


@dataclass
class RecordedOrder:
    order_id: str
    timestamp: datetime
    symbol: str
    side: str
    qty: int
    order_type: str
    price: float
    trigger_price: Optional[float] = None
    status: str = "PLACED"
    reason: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class TradeReport:
    trade_id: str
    symbol: str
    option_type: str
    strike: float
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: int
    gross_pnl: float
    brokerage: float
    net_pnl: float
    deployed_capital: float
    max_favorable_price: float
    trailing_activated: bool
    exit_reason: str
    holding_minutes: float
    roi_pct: float

    def to_dict(self):
        return asdict(self)


class FillModel:
    """Turns an intended action into a realistic fill price against the book."""

    @staticmethod
    def round_tick(price: float, up: bool) -> float:
        t = Config.TICK_SIZE
        if price <= 0:
            return t
        steps = price / t
        steps = np.ceil(steps) if up else np.floor(steps)
        return max(t, round(steps * t, 2))

    @staticmethod
    def entry_buy(depth: MarketDepth) -> float:
                                        
        return FillModel.round_tick(depth.best_ask or depth.mid, up=True)

    @staticmethod
    def exit_sell(depth: MarketDepth) -> float:
                                  
        return FillModel.round_tick(depth.best_bid or depth.mid, up=False)

    @staticmethod
    def stop_fill(trigger: float) -> float:
                                                                     
        slip = Config.STOP_SLIPPAGE_TICKS * Config.TICK_SIZE
        return FillModel.round_tick(max(Config.TICK_SIZE, trigger - slip), up=False)


class TrendAnalyzer:
    def __init__(self, lookback=50):
        self.price = deque(maxlen=lookback)
        self.vol = deque(maxlen=lookback)

    def update(self, price: float, volume: int):
        if price > 0:
            self.price.append(price)
            self.vol.append(max(0, volume))

    def signal(self) -> Dict:
        if len(self.price) < 20:
            return {"trend": "NEUTRAL", "strength": 0.0, "vwap": 0.0,
                    "current_vs_vwap": 0.0, "volume_ratio": 1.0}
        p = np.array(self.price)
        v = np.array(self.vol)
        vwap = float(np.sum(p * v) / np.sum(v)) if np.sum(v) > 0 else float(p[-1])
        slope = float(np.polyfit(np.arange(len(p)), p, 1)[0])
        std = float(np.std(p)) or 1e-9
        strength = abs(slope) / std
        if slope > std * 0.01:
            trend = "BULLISH"
        elif slope < -std * 0.01:
            trend = "BEARISH"
        else:
            trend = "NEUTRAL"
        recent = np.mean(v[-5:]) if len(v) >= 5 else np.mean(v)
        older = np.mean(v[:-5]) if len(v) > 5 else recent
        vol_ratio = float(recent / older) if older > 0 else 1.0
        if vol_ratio > 1.2:
            strength *= 1.5
        elif vol_ratio < 0.8:
            strength *= 0.5
        return {"trend": trend, "strength": min(1.0, strength), "vwap": vwap,
                "current_vs_vwap": (p[-1] - vwap) / vwap if vwap else 0.0,
                "volume_ratio": vol_ratio}


class Position:
    def __init__(self, symbol, entry_price, qty, entry_time,
                 option_type, strike, deployed_capital, recorder):
        self.trade_id = f"{now_ist():%Y%m%d}_{int(time.time()*1000)}"
        self.symbol = symbol
        self.entry_price = entry_price
        self.qty = qty
        self.entry_time = entry_time
        self.option_type = option_type
        self.strike = strike
        self.deployed_capital = deployed_capital
        self.recorder = recorder

        self.stop_loss: Optional[float] = None
        self.original_stop: Optional[float] = None
        self.target: Optional[float] = None
        self.max_favorable_price = entry_price

        self.is_trailing_active = False
        self.target_removed = False
        self.trailing_level = 0
        self.stop_order_ids: List[str] = []
        self.entry_order_id: Optional[str] = None

                                                                               
    def set_initial_stops(self, target: float):
                                                                                 
        risk_per_share = Config.MAX_LOSS_PER_TRADE / self.qty
        sanity_floor = self.entry_price * (1 - Config.STOP_SANITY_FLOOR_PCT)
        self.stop_loss = max(self.entry_price - risk_per_share, sanity_floor)
        self.original_stop = self.stop_loss
        self.target = target
        log.info(f"{self.trade_id} SETUP  entry=₹{self.entry_price:.2f} "
                 f"stop=₹{self.stop_loss:.2f} target=₹{target:.2f} "
                 f"(risk ≈₹{(self.entry_price - self.stop_loss) * self.qty:.0f})")

                                                                               
    def update_trailing(self, current_price: float) -> bool:
        profit = (current_price - self.entry_price) * self.qty
        if current_price > self.max_favorable_price:
            self.max_favorable_price = current_price
        if profit < Config.TRAILING_START_PROFIT:
            return False

        if not self.target_removed:
            self.target_removed = True
            self.target = None
            log.info(f"{self.trade_id} target removed — trailing engaged")

        level = max(0, int((profit - Config.TRAILING_START_PROFIT) / Config.TRAILING_STEP))
        locked_profit = Config.TRAILING_START_PROFIT + level * Config.TRAILING_STEP
        new_stop = self.entry_price + locked_profit / self.qty

        if not self.is_trailing_active or level > self.trailing_level:
            self.is_trailing_active = True
            self.trailing_level = level
            self.stop_loss = new_stop
            self._record_stop(new_stop, "trailing stop")
            log.info(f"{self.trade_id} TRAIL level={level} "
                     f"stop=₹{new_stop:.2f} (locks ₹{locked_profit:.0f})")
            return True
        return False

    def _record_stop(self, stop_price: float, reason: str):
        if self.stop_order_ids:
            self.recorder.record(RecordedOrder(
                order_id=f"CANCEL_{self.stop_order_ids[-1]}", timestamp=now_ist(),
                symbol=self.symbol, side="CANCEL", qty=0, order_type="CANCEL",
                price=0, status="CANCELLED", reason="replaced by new stop"))
        trig = FillModel.round_tick(stop_price, up=False)
        oid = f"SL_{self.trade_id}_{len(self.stop_order_ids)+1}"
        self.recorder.record(RecordedOrder(
            order_id=oid, timestamp=now_ist(), symbol=self.symbol, side="SELL",
            qty=self.qty, order_type="SL", price=trig - Config.TICK_SIZE,
            trigger_price=trig, status="PLACED", reason=reason))
        self.stop_order_ids.append(oid)

                                                                               
    def hit_stop(self, price: float) -> bool:
        return price <= self.stop_loss

    def hit_target(self, price: float) -> bool:
        return self.target is not None and price >= self.target

    def close(self, exit_price: float, exit_time: datetime, reason: str) -> TradeReport:
        gross = (exit_price - self.entry_price) * self.qty
        lots = self.qty / Config.LOT_SIZE
        brokerage = Config.BROKERAGE_PER_LOT_PER_SIDE * 2 * lots
        net = gross - brokerage
        mins = (exit_time - self.entry_time).total_seconds() / 60
        roi = (net / self.deployed_capital * 100) if self.deployed_capital else 0.0
        self.recorder.record(RecordedOrder(
            order_id=f"EXIT_{self.trade_id}", timestamp=exit_time, symbol=self.symbol,
            side="SELL", qty=self.qty, order_type="MARKET", price=exit_price,
            status="EXECUTED", reason=reason))
        return TradeReport(
            trade_id=self.trade_id, symbol=self.symbol, option_type=self.option_type,
            strike=self.strike, entry_time=self.entry_time, exit_time=exit_time,
            entry_price=self.entry_price, exit_price=exit_price, quantity=self.qty,
            gross_pnl=gross, brokerage=brokerage, net_pnl=net,
            deployed_capital=self.deployed_capital,
            max_favorable_price=self.max_favorable_price,
            trailing_activated=self.is_trailing_active, exit_reason=reason,
            holding_minutes=mins, roi_pct=roi)


class OrderRecorder:
    def __init__(self):
        self.orders: List[RecordedOrder] = []

    def record(self, order: RecordedOrder):
        self.orders.append(order)
        log.info(f"ORDER #{len(self.orders)}: {order.side} {order.qty} "
                 f"{order.symbol} @ ₹{order.price:.2f} [{order.order_type}]")

    def summary(self) -> Dict:
        return {"total": len(self.orders),
                "entries": sum(o.side == "BUY" for o in self.orders),
                "exits": sum(o.side == "SELL" and o.order_type != "SL" for o in self.orders),
                "stops": sum(o.order_type == "SL" for o in self.orders),
                "cancels": sum(o.side == "CANCEL" for o in self.orders)}


class Metrics:
    def __init__(self, capital: float):
        self.capital = capital
        self.trades: List[TradeReport] = []

    def add(self, t: TradeReport):
        self.trades.append(t)

    def compute(self) -> Dict:
        if not self.trades:
            return {"trades": 0, "wins": 0, "losses": 0, "breakeven": 0,
                    "win_rate": 0.0, "net_pnl": 0.0, "gross_pnl": 0.0,
                    "brokerage": 0.0, "roi": 0.0, "profit_factor": 0.0,
                    "avg_win": 0.0, "avg_loss": 0.0, "max_win": 0.0, "max_loss": 0.0}
        wins = [t for t in self.trades if t.net_pnl > 0]
        losses = [t for t in self.trades if t.net_pnl < 0]
        be = [t for t in self.trades if t.net_pnl == 0]
        gp = sum(t.net_pnl for t in wins)
        gl = abs(sum(t.net_pnl for t in losses))
        net = sum(t.net_pnl for t in self.trades)
        decided = len(wins) + len(losses)
        return {
            "trades": len(self.trades), "wins": len(wins),
            "losses": len(losses), "breakeven": len(be),
            "win_rate": (len(wins) / decided * 100) if decided else 0.0,
            "net_pnl": net, "gross_pnl": sum(t.gross_pnl for t in self.trades),
            "brokerage": sum(t.brokerage for t in self.trades),
            "roi": net / self.capital * 100 if self.capital else 0.0,
            "profit_factor": (gp / gl) if gl else float("inf"),
            "avg_win": (gp / len(wins)) if wins else 0.0,
            "avg_loss": (-gl / len(losses)) if losses else 0.0,
            "max_win": max((t.net_pnl for t in wins), default=0.0),
            "max_loss": min((t.net_pnl for t in losses), default=0.0)}


class NiftyOptionsEngine:
    def __init__(self, capital: float):
        self.capital = capital
        self.margin_used = 0.0
        self.spot_price = 0.0

        self.kite = None
        api, tok = os.getenv("API_KEY"), os.getenv("ACCESS_TOKEN")
        if HAVE_KITE and api and tok:
            self.kite = KiteConnect(api_key=api)
            self.kite.set_access_token(tok)

        self.positions: Dict[int, Position] = {}
        self.recorder = OrderRecorder()
        self.metrics = Metrics(capital)

                                                         
        self.trend: Dict[int, TrendAnalyzer] = {}
        self.last_traded_volume: Dict[int, int] = {}

        self.monitored: Dict[int, Dict] = {}
        self.depths: Dict[int, MarketDepth] = {}
        self.last_signal_time: Dict[int, datetime] = {}
        self.last_trade_time: Dict[int, datetime] = {}

        self.tick_q: "queue.Queue" = queue.Queue()
        self.kws = None
        self.running = False
        self.tick_count = 0
        self.trades_today = 0
        self.halted = False

        log.info(f"Engine init | capital ₹{capital:,.0f} | "
                 f"Kite: {'connected' if self.kite else 'OFFLINE'}")

                                                                               
    def utilization(self) -> float:
        return self.margin_used / self.capital if self.capital else 0.0

    def in_focus_mode(self) -> bool:
        return self.utilization() >= Config.FOCUS_MODE_THRESHOLD

    @staticmethod
    def market_open() -> bool:
        n = now_ist()
        o = n.replace(hour=Config.MKT_OPEN[0], minute=Config.MKT_OPEN[1],
                      second=0, microsecond=0)
        c = n.replace(hour=Config.MKT_CLOSE[0], minute=Config.MKT_CLOSE[1],
                      second=0, microsecond=0)
        return o <= n <= c

    def check_circuit_breaker(self):
        if self.halted:
            return
        if self.metrics.compute()["net_pnl"] <= -Config.MAX_DAILY_LOSS:
            self.halted = True
            log.error(f"CIRCUIT BREAKER: daily loss ≥ ₹{Config.MAX_DAILY_LOSS:,.0f} "
                      f"— no new entries this session")

    def position_size(self, confidence: float, entry_price: float) -> Tuple[int, float]:
        available = self.capital * Config.MAX_CAPITAL_UTILIZATION - self.margin_used
        if self.in_focus_mode() or available <= 0:
            return 0, 0.0
        per_lot_cost = entry_price * Config.LOT_SIZE
        if per_lot_cost > available:
            return 0, 0.0
        lots = min(Config.MAX_OPEN_POSITIONS,
                   int(min(1.0, confidence * 1.2) * 2) + 1,
                   int(available / per_lot_cost))
        lots = max(0, lots)
        qty = lots * Config.LOT_SIZE
        return qty, qty * entry_price

    def worth_trading(self, entry, qty, target, stop) -> Dict:
        lots = qty / Config.LOT_SIZE
        brk = Config.BROKERAGE_PER_LOT_PER_SIDE * 2
        net_profit_per_lot = abs(target - entry) * Config.LOT_SIZE - brk
        net_loss_per_lot = abs(entry - stop) * Config.LOT_SIZE + brk
        rr = net_profit_per_lot / net_loss_per_lot if net_loss_per_lot else 0
        ok = (net_profit_per_lot >= Config.MIN_NET_PROFIT_PER_LOT
              and rr >= Config.MIN_RISK_REWARD)
        return {"ok": ok, "rr": rr,
                "net_profit_potential": net_profit_per_lot * lots}

                                                                               
    def get_tradeable_options(self) -> List[Dict]:
        if not self.kite:
            return []
        try:
            q = self.kite.quote(["NSE:NIFTY 50"])
            self.spot_price = q["NSE:NIFTY 50"]["last_price"]
            log.info(f"NIFTY spot ₹{self.spot_price:.2f}")
            atm = round(self.spot_price / 50) * 50
            inst = pd.DataFrame(self.kite.instruments("NFO"))
            opts = inst[(inst["name"] == "NIFTY")
                        & (inst["strike"] >= atm - 150)
                        & (inst["strike"] <= atm + 150)].copy()
            opts["expiry_dt"] = pd.to_datetime(opts["expiry"])
            today = now_ist().date()
            future = sorted(opts[opts["expiry_dt"].dt.date >= today]["expiry_dt"].unique())
            weekly = next((e for e in future
                           if (pd.Timestamp(e).date() - today).days <= 7), future[0])
            sel = opts[opts["expiry_dt"] == weekly].sort_values(["strike", "instrument_type"])
            return sel.to_dict("records")
        except Exception as e:
            log.error(f"get_tradeable_options: {e}")
            return []

    @staticmethod
    def _depth_from_tick(tick: Dict) -> Optional[MarketDepth]:
        d = tick.get("depth")
        if not d or not d.get("buy") or not d.get("sell"):
            return None
        bids = [(x["price"], x["quantity"], x.get("orders", 0))
                for x in d["buy"][:5] if x.get("price", 0) > 0]
        asks = [(x["price"], x["quantity"], x.get("orders", 0))
                for x in d["sell"][:5] if x.get("price", 0) > 0]
        if not bids or not asks:
            return None
        return MarketDepth(time.time(), bids, asks)

                                                                               
    def open_trade(self, token, depth: MarketDepth, confidence: float):
        info = self.monitored[token]
        entry = FillModel.entry_buy(depth)
        qty, deployed = self.position_size(confidence, entry)
        if qty <= 0:
            return
        target = FillModel.round_tick(entry * Config.TARGET_MULTIPLIER, up=False)

                                                             
        risk_ps = Config.MAX_LOSS_PER_TRADE / qty
        provisional_stop = max(entry - risk_ps, entry * (1 - Config.STOP_SANITY_FLOOR_PCT))
        wt = self.worth_trading(entry, qty, target, provisional_stop)
        if not wt["ok"]:
            log.info(f"{info['symbol']} rejected — RR {wt['rr']:.2f} / "
                     f"net ₹{wt['net_profit_potential']:.0f}")
            return

        self.recorder.record(RecordedOrder(
            order_id=f"ENTRY_{self.trades_today+1}_{int(time.time())}",
            timestamp=now_ist(), symbol=info["symbol"], side="BUY", qty=qty,
            order_type="LIMIT", price=entry, status="EXECUTED", reason="entry"))

        pos = Position(info["symbol"], entry, qty, now_ist(),
                       info["type"], info["strike"], deployed, self.recorder)
        pos.set_initial_stops(target)
        pos._record_stop(pos.stop_loss, "initial stop")

        self.positions[token] = pos
        self.margin_used += deployed
        self.trades_today += 1
        self.last_trade_time[token] = now_ist()
        log.info(f"OPEN {info['symbol']} qty={qty} entry=₹{entry:.2f} "
                 f"conf={confidence:.2f} util={self.utilization():.0%}")

    def close_trade(self, token, exit_price, reason):
        pos = self.positions.get(token)
        if not pos:
            return
        report = pos.close(exit_price, now_ist(), reason)
        self.metrics.add(report)
        self.margin_used -= pos.deployed_capital
        del self.positions[token]
        m = self.metrics.compute()
        log.info(f"CLOSE {pos.symbol} [{reason}] net=₹{report.net_pnl:.0f} "
                 f"| session ₹{m['net_pnl']:.0f} win%={m['win_rate']:.0f}")
        self.check_circuit_breaker()

                                                                               
    def manage_open_position(self, token, depth: MarketDepth):
        pos = self.positions.get(token)
        if not pos:
            return
                                                                        
        px = depth.best_bid or depth.mid
        pos.update_trailing(px)
        if pos.hit_stop(px):
            self.close_trade(token, FillModel.stop_fill(pos.stop_loss), "STOP_LOSS")
        elif pos.hit_target(px):
            self.close_trade(token, FillModel.exit_sell(depth), "TARGET")

    def evaluate_entry(self, token, depth: MarketDepth):
        if self.halted or not self.market_open() or self.in_focus_mode():
            return
        if token in self.positions:
            return
        if (len(self.positions) >= Config.MAX_OPEN_POSITIONS
                or self.trades_today >= Config.MAX_DAILY_TRADES):
            return
        mid = depth.mid
        if mid <= 0 or depth.spread > mid * Config.MAX_SPREAD_PCT:
            return
        if depth.total_depth_volume < Config.MIN_DEPTH_VOLUME:
            return

        now = now_ist()
        if now - self.last_signal_time.get(token, datetime.min.replace(
                tzinfo=now.tzinfo)) < timedelta(seconds=Config.SIGNAL_COOLDOWN_S):
            return
        if now - self.last_trade_time.get(token, datetime.min.replace(
                tzinfo=now.tzinfo)) < timedelta(seconds=Config.TRADE_COOLDOWN_S):
            self.last_signal_time[token] = now
            return

        trend = self.trend[token].signal()
        imb = depth.volume_imbalance
        confidence = 0.0

                                                                
        if imb > 0.35:
            confidence = min(0.85, 0.65 + abs(imb) * 0.4)
                                                                              
        elif (imb > 0.25 and trend["trend"] == "BULLISH"
              and trend["strength"] >= 0.4 and trend["volume_ratio"] > 1.3):
            confidence = min(0.80, 0.60 + trend["strength"] * 0.3)

        self.last_signal_time[token] = now
        if confidence >= Config.MIN_CONFIDENCE:
            log.info(f"SIGNAL {self.monitored[token]['symbol']} "
                     f"imb={imb:+.0%} trend={trend['trend']} conf={confidence:.2f}")
            self.open_trade(token, depth, confidence)

                                                                               
    async def stream(self):
        opts = self.get_tradeable_options()
        if not opts:
            log.warning("no tradeable options")
            return
        atm = round(self.spot_price / 50) * 50
        selected = [o for o in opts if abs(o["strike"] - atm) <= 150][:10]
        for o in selected:
            tok = o["instrument_token"]
            self.monitored[tok] = {"symbol": o["tradingsymbol"], "strike": o["strike"],
                                   "type": o["instrument_type"], "expiry": o["expiry_dt"]}
            self.trend[tok] = TrendAnalyzer()
        tokens = [o["instrument_token"] for o in selected]
        log.info(f"monitoring {len(tokens)} options")

        if not self.kite:
            log.error("no Kite connection — cannot stream")
            return

        self.kws = KiteTicker(os.getenv("API_KEY"), os.getenv("ACCESS_TOKEN"))
        self.kws.on_ticks = lambda ws, ticks: [self.tick_q.put(t) for t in ticks]

        def on_connect(ws, _):
            ws.subscribe(tokens)
            ws.set_mode(ws.MODE_FULL, tokens)
            log.info("websocket connected")

        def on_close(ws, code, reason):
            log.warning(f"websocket closed {code} {reason} — will retry")

        self.kws.on_connect = on_connect
        self.kws.on_close = on_close
                                                                                   
        try:
            self.kws.on_reconnect = lambda ws, n: log.warning(f"reconnect attempt {n}")
        except Exception:
            pass

        threading.Thread(target=self._run_ws, daemon=True).start()
        await self._consume()

    def _run_ws(self):
        try:
            self.kws.connect(threaded=True)
        except Exception as e:
            log.error(f"ws thread: {e}")

    async def _consume(self):
        while self.running:
            try:
                tick = self.tick_q.get(timeout=0.1)
            except queue.Empty:
                await asyncio.sleep(0.05)
                continue
            try:
                self.tick_count += 1
                token = tick.get("instrument_token")
                if token not in self.monitored:
                    continue

                                                                               
                ltp = tick.get("last_price", 0)
                vtraded = tick.get("volume_traded", 0)
                prev = self.last_traded_volume.get(token, vtraded)
                vol_delta = max(0, vtraded - prev)
                self.last_traded_volume[token] = vtraded
                if ltp > 0:
                    self.trend[token].update(ltp, vol_delta or 1)

                depth = self._depth_from_tick(tick)
                if depth:
                    self.depths[token] = depth
                    if token in self.positions:
                        self.manage_open_position(token, depth)
                    if self.tick_count % Config.SIGNAL_EVERY_N_TICKS == 0:
                        self.evaluate_entry(token, depth)

                if self.tick_count % 200 == 0:
                    self._dashboard()
            except Exception as e:
                log.error(f"tick error: {e}")
                await asyncio.sleep(0.05)

                                                                               
    def _dashboard(self):
        m = self.metrics.compute()
        log.info(f"[{now_ist():%H:%M:%S}] ticks={self.tick_count} "
                 f"open={len(self.positions)} trades={m['trades']} "
                 f"net=₹{m['net_pnl']:.0f} win%={m['win_rate']:.0f} "
                 f"util={self.utilization():.0%} "
                 f"{'HALTED' if self.halted else ''}")

    def write_report(self):
        reports = Path("reports")
        reports.mkdir(exist_ok=True)
        m = self.metrics.compute()
        payload = {
            "generated": now_ist().isoformat(),
            "mode": "live-data simulation",
            "capital": self.capital,
            "metrics": m,
            "orders": self.recorder.summary(),
            "config": {"max_loss_per_trade": Config.MAX_LOSS_PER_TRADE,
                       "max_daily_loss": Config.MAX_DAILY_LOSS,
                       "trailing_start": Config.TRAILING_START_PROFIT,
                       "trailing_step": Config.TRAILING_STEP,
                       "stop_slippage_ticks": Config.STOP_SLIPPAGE_TICKS},
            "trades": [t.to_dict() for t in self.metrics.trades],
            "ticks_processed": self.tick_count,
            "disclaimer": ("Simulated fills against live tick data. Optimistic vs a "
                           "real exchange; not a live track record."),
        }
        path = reports / f"session_{now_ist():%Y-%m-%d}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        log.info(f"report -> {path}")

                                                                               
    async def start(self):
        if not (os.getenv("API_KEY") and os.getenv("ACCESS_TOKEN")):
            log.error("API_KEY / ACCESS_TOKEN not set")
            return
        self.running = True
        log.info("starting (live-data simulation)…")
        await self.stream()

    async def shutdown(self):
        self.running = False
        if self.kws:
            try:
                self.kws.close()
            except Exception:
                pass
        for token in list(self.positions):
            depth = self.depths.get(token)
            px = depth.best_bid if depth else self.positions[token].entry_price
            self.close_trade(token, px, "SHUTDOWN")
        self.write_report()
        m = self.metrics.compute()
        log.info(f"FINAL trades={m['trades']} win%={m['win_rate']:.0f} "
                 f"net=₹{m['net_pnl']:.0f} roi={m['roi']:.2f}%")


async def _main():
    capital = float(os.getenv("TRADING_CAPITAL", "100000"))
    engine = NiftyOptionsEngine(capital)

    def _sig(_s, _f):
        log.info("signal received — shutting down")
        asyncio.get_event_loop().create_task(engine.shutdown())
    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(s, _sig)
        except Exception:
            pass

    try:
        await engine.start()
    except KeyboardInterrupt:
        pass
    finally:
        await engine.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        log.info("interrupted")
