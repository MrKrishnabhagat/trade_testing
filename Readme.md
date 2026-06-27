# NIFTY Options Intraday Engine

An event-driven intraday trading engine for NIFTY options. It consumes live
order-book data from the Zerodha Kite Connect WebSocket and runs a full trade
lifecycle — signal generation, entry, risk-budgeted stop placement, profit-based
trailing, and exit — in a **simulation mode** that records every order instead of
sending it to the exchange.

> **Mode & scope.** This runs against *live* tick data but does not place real
> orders. Fills are modelled against the order book with slippage, which is more
> realistic than filling at mid, but still optimistic versus a real exchange
> (no queue position, partial fills, or rejection on fast moves). The output is a
> logic-validation harness, not a live trading track record.

## Architecture

The engine is a single asyncio process with a producer/consumer split:

- A background thread receives ticks from the Kite WebSocket and pushes them onto
  a queue.
- The async consumer drains the queue and, per tick, updates the per-instrument
  trend state, manages any open position, and (every *N* ticks) evaluates entries.

Decoupling the socket thread from the trading logic via the queue keeps tick
ingestion from blocking on order/exit processing.

Core components:

| Component | Responsibility |
|---|---|
| `NiftyOptionsEngine` | Orchestration: streaming, signal evaluation, position management, reporting |
| `Position` | Per-trade state, the risk-budgeted stop, and the profit-based trailing ladder |
| `TrendAnalyzer` | Per-instrument VWAP / slope / volume-ratio trend signal |
| `FillModel` | Converts intended actions into realistic fills (cross the spread, slip stops) |
| `BlackScholes` | Option pricing, Greeks, and Newton-Raphson implied volatility |
| `OrderRecorder` / `Metrics` | Order journal and post-run performance metrics |

## Signal logic

Entries are **long-only** and gated before any signal is considered: market open,
spread under 2% of price, sufficient book depth, position/daily-trade limits, and
per-instrument cooldowns must all pass first.

Two entry triggers, checked in order:

1. **Order-book imbalance.** When top-of-book bid volume exceeds ask volume past a
   threshold (`imbalance > 0.35`), confidence scales with how lopsided the book is.
2. **Imbalance with trend confirmation.** A milder imbalance (`> 0.25`) is taken
   only when the instrument's own price trend reads bullish on rising volume.

A trade fires only if confidence clears `MIN_CONFIDENCE` (0.65).

## Risk management

- **Per-trade stop as a rupee budget.** The initial stop is sized so the trade
  risks at most `MAX_LOSS_PER_TRADE` (default ₹600), floored by a wide sanity cap
  — a genuine risk-based stop rather than a fixed percentage.
- **Profit-based trailing.** Once unrealised profit reaches ₹150, the fixed target
  is dropped and the stop ratchets up in ₹100 profit-steps, locking in progressively
  more profit and never moving down. On a reversal, the position exits at the
  locked level.
- **Portfolio circuit breaker.** A session daily-loss limit (`MAX_DAILY_LOSS`)
  latches and halts all new entries once breached.
- **Capital controls.** Utilisation is capped at 80%, with a focus mode at 70% that
  stops opening new trades and manages existing ones only.
- **Realistic fills.** Entries pay the ask, exits hit the bid, and protective stops
  slip *through* the trigger by a configurable number of ticks.

## Setup

```bash
pip install -r requirements.txt
```

Set Kite Connect credentials as environment variables:

```bash
export API_KEY="your_kite_api_key"
export ACCESS_TOKEN="your_kite_access_token"
export TRADING_CAPITAL="100000"        # optional, default ₹1,00,000
```

## Run

```bash
python nifty_options_engine.py
```

During market hours the engine subscribes to ~10 near-ATM weekly options, streams
full-depth ticks, and prints a periodic dashboard. On shutdown it flattens open
positions and writes a session report.

## Configuration

Key parameters live in the `Config` class; several are overridable via environment
variables:

| Variable | Default | Meaning |
|---|---|---|
| `TRADING_CAPITAL` | 100000 | Starting capital (₹) |
| `MAX_LOSS_PER_TRADE` | 600 | Per-trade risk budget (₹) |
| `MAX_DAILY_LOSS` | 2500 | Session loss limit before halting (₹) |
| `MAX_DAILY_TRADES` | 10 | Daily trade cap |

Trailing thresholds, capital-utilisation caps, cooldowns, and the spread/volume
filters are set in `Config` directly.

## Output

Each run produces:

- `logs/trading_<date>.log` — full event log
- `reports/session_<date>.json` — metrics, order summary, per-trade detail, and the
  config used

## Project layout

```
nifty_options_engine.py    # the engine
requirements.txt
README.md
logs/                      # created at runtime
reports/                   # created at runtime
```
