"""Web dashboard for the paper-trading bot.

Single-process aiohttp server. Reads from the live broker for cash/positions
(no DB lock contention) and from SQLite for historical fills/resolutions/NAV.
Serves a self-contained HTML page at / and JSON endpoints under /api.

Run as a supervised task in main; default port 8080. Open
http://localhost:8080 to see portfolio value + asset breakdowns + recent
fills + NAV trajectory + per-strategy P&L.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from typing import Any

import structlog
from aiohttp import web

from polyagent.config import settings
from polyagent.gamma import Market
from polyagent.orderbook import BookStore
from polyagent.paper_broker import PaperBroker

log = structlog.get_logger()


def _read_only_conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=10000")
    return conn


class Dashboard:
    def __init__(
        self,
        broker: PaperBroker,
        book_store: BookStore,
        markets: list[Market],
        db_path: str,
        port: int = 8080,
    ) -> None:
        self.broker = broker
        self.book_store = book_store
        self.markets_by_token = {m.yes_token_id: m for m in markets}
        self.markets_by_token.update({m.no_token_id: m for m in markets})
        self.db_path = db_path
        self.port = port

    # ─────────── data layer ───────────
    def _summary(self) -> dict[str, Any]:
        s = self.broker.summary()
        # Per-strategy realized P&L over all time (joined to resolutions).
        try:
            conn = _read_only_conn(self.db_path)
            rows = conn.execute(
                """
                SELECT f.strategy,
                       COUNT(*) AS n,
                       SUM(CASE
                           WHEN (f.token_id=r.yes_token_id AND r.yes_won=1)
                             OR (f.token_id=r.no_token_id AND r.yes_won=0) THEN 1
                           ELSE 0 END) AS wins,
                       ROUND(SUM(((CASE
                           WHEN (f.token_id=r.yes_token_id AND r.yes_won=1)
                             OR (f.token_id=r.no_token_id AND r.yes_won=0)
                                THEN 1.0 ELSE 0.0 END) - f.price) * f.size), 2) AS pnl,
                       ROUND(SUM(f.notional), 2) AS notional
                FROM fills f
                INNER JOIN resolutions r ON f.condition_id = r.condition_id
                GROUP BY f.strategy
                """
            ).fetchall()
            strategies = [
                {
                    "strategy": r["strategy"],
                    "resolved": r["n"],
                    "wins": r["wins"] or 0,
                    "win_rate": (r["wins"] or 0) / r["n"] if r["n"] else 0.0,
                    "realized_pnl": r["pnl"] or 0.0,
                    "notional": r["notional"] or 0.0,
                }
                for r in rows
            ]
            # Per-strategy unresolved fills count + notional
            urows = conn.execute(
                """
                SELECT f.strategy,
                       COUNT(*) AS n,
                       ROUND(SUM(f.notional), 2) AS notional
                FROM fills f
                LEFT JOIN resolutions r ON f.condition_id = r.condition_id
                WHERE r.condition_id IS NULL
                GROUP BY f.strategy
                """
            ).fetchall()
            unresolved = {
                r["strategy"]: {"open": r["n"], "open_notional": r["notional"] or 0.0}
                for r in urows
            }
            for s_row in strategies:
                u = unresolved.pop(s_row["strategy"], None)
                if u:
                    s_row.update(u)
                else:
                    s_row.update({"open": 0, "open_notional": 0.0})
            for name, u in unresolved.items():
                strategies.append(
                    {
                        "strategy": name,
                        "resolved": 0,
                        "wins": 0,
                        "win_rate": 0.0,
                        "realized_pnl": 0.0,
                        "notional": 0.0,
                        **u,
                    }
                )
            conn.close()
        except Exception as e:
            log.warning("dashboard_strategy_query_error", err=str(e))
            strategies = []
        return {
            "summary": s,
            "strategies": strategies,
            "ts": time.time(),
        }

    def _positions(self) -> list[dict[str, Any]]:
        out = []
        for p in self.broker.positions.values():
            if p.size <= 0:
                continue
            book = self.book_store.books.get(p.token_id)
            bid = book.best_bid() if book else None
            ask = book.best_ask() if book else None
            mid = book.mid() if book else None
            mark_bid = bid[0] if bid else (mid or p.avg_cost)
            mark_mid = mid if mid is not None else (bid[0] if bid else p.avg_cost)
            market = self.markets_by_token.get(p.token_id)
            side = "?"
            if market:
                if p.token_id == market.yes_token_id:
                    side = "YES"
                elif p.token_id == market.no_token_id:
                    side = "NO"
            out.append(
                {
                    "token_id": p.token_id,
                    "condition_id": market.condition_id if market else None,
                    "question": market.question if market else None,
                    "category": market.category if market else None,
                    "side": side,
                    "size": round(p.size, 2),
                    "avg_cost": round(p.avg_cost, 4),
                    "bid": round(bid[0], 4) if bid else None,
                    "ask": round(ask[0], 4) if ask else None,
                    "mid": round(mid, 4) if mid is not None else None,
                    "value_bid": round(p.size * mark_bid, 2),
                    "value_mid": round(p.size * mark_mid, 2),
                    "unrealized_bid": round((mark_bid - p.avg_cost) * p.size, 2),
                    "unrealized_mid": round((mark_mid - p.avg_cost) * p.size, 2),
                }
            )
        out.sort(key=lambda r: -abs(r["value_mid"] or 0))
        return out

    def _recent_fills(self, limit: int = 50) -> list[dict[str, Any]]:
        try:
            conn = _read_only_conn(self.db_path)
            rows = conn.execute(
                """
                SELECT ts, strategy, condition_id, token_id, side, price, size, notional, reason
                FROM fills ORDER BY ts DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
            conn.close()
        except Exception as e:
            log.warning("dashboard_fills_query_error", err=str(e))
            return []
        out = []
        for r in rows:
            market = self.markets_by_token.get(r["token_id"])
            yes_no = "?"
            if market:
                if r["token_id"] == market.yes_token_id:
                    yes_no = "YES"
                elif r["token_id"] == market.no_token_id:
                    yes_no = "NO"
            out.append(
                {
                    "ts": r["ts"],
                    "ts_iso": time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.gmtime(r["ts"])
                    ),
                    "strategy": r["strategy"],
                    "side": r["side"],
                    "yes_no": yes_no,
                    "question": market.question if market else r["condition_id"][:16] + "..",
                    "price": r["price"],
                    "size": r["size"],
                    "notional": r["notional"],
                }
            )
        return out

    def _nav_history(self, limit: int = 500) -> list[dict[str, Any]]:
        try:
            conn = _read_only_conn(self.db_path)
            rows = conn.execute(
                """
                SELECT ts, cash, position_value, nav
                FROM nav_history ORDER BY ts ASC
                LIMIT (SELECT MAX(0, COUNT(*) - ?) FROM nav_history) OFFSET 0
                """,
                (limit,),
            ).fetchall()
            # SQLite weirdness above — simpler: just take last N.
            rows = conn.execute(
                "SELECT ts, cash, position_value, nav FROM nav_history ORDER BY ts DESC LIMIT ?",
                (limit,),
            ).fetchall()
            conn.close()
        except Exception as e:
            log.warning("dashboard_nav_query_error", err=str(e))
            return []
        return list(reversed([dict(r) for r in rows]))

    def _recent_resolutions(self, limit: int = 25) -> list[dict[str, Any]]:
        try:
            conn = _read_only_conn(self.db_path)
            rows = conn.execute(
                """
                SELECT condition_id, resolved_ts, yes_won, yes_size, no_size, pnl, detail
                FROM resolutions
                WHERE COALESCE(yes_size,0) > 0 OR COALESCE(no_size,0) > 0
                ORDER BY resolved_ts DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
            conn.close()
        except Exception as e:
            log.warning("dashboard_res_query_error", err=str(e))
            return []
        out = []
        for r in rows:
            try:
                d = json.loads(r["detail"] or "{}")
            except json.JSONDecodeError:
                d = {}
            out.append(
                {
                    "ts": r["resolved_ts"],
                    "ts_iso": time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.gmtime(r["resolved_ts"])
                    ),
                    "yes_won": bool(r["yes_won"]),
                    "yes_size": r["yes_size"],
                    "no_size": r["no_size"],
                    "pnl": r["pnl"],
                    "question": d.get("question", ""),
                }
            )
        return out

    # ─────────── handlers ───────────
    async def api_summary(self, request: web.Request) -> web.Response:
        return web.json_response(self._summary())

    async def api_positions(self, request: web.Request) -> web.Response:
        return web.json_response(self._positions())

    async def api_fills(self, request: web.Request) -> web.Response:
        limit = int(request.query.get("limit", "50"))
        return web.json_response(self._recent_fills(limit))

    async def api_nav_history(self, request: web.Request) -> web.Response:
        limit = int(request.query.get("limit", "500"))
        return web.json_response(self._nav_history(limit))

    async def api_resolutions(self, request: web.Request) -> web.Response:
        limit = int(request.query.get("limit", "25"))
        return web.json_response(self._recent_resolutions(limit))

    async def api_pessimistic_nav(self, request: web.Request) -> web.Response:
        """Compute realized P&L using shadow ledger pessimistic prices.

        Joins fills + fills_shadow + resolutions to figure out:
        - what we paid (pessimistic_price × size)
        - what we received at settle (1.0 × size if winning leg, else 0)
        Returns the parallel pessimistic P&L for resolved fills only.
        """
        try:
            conn = _read_only_conn(self.db_path)
            rows = conn.execute(
                """
                SELECT
                    f.strategy,
                    SUM(((CASE WHEN (f.token_id=r.yes_token_id AND r.yes_won=1)
                                OR (f.token_id=r.no_token_id AND r.yes_won=0)
                              THEN 1.0 ELSE 0.0 END) - COALESCE(s.pessimistic_price, f.price)) * f.size) AS pess_pnl,
                    SUM(((CASE WHEN (f.token_id=r.yes_token_id AND r.yes_won=1)
                                OR (f.token_id=r.no_token_id AND r.yes_won=0)
                              THEN 1.0 ELSE 0.0 END) - f.price) * f.size) AS realized_pnl,
                    COUNT(*) AS n
                FROM fills f
                INNER JOIN resolutions r ON f.condition_id = r.condition_id
                LEFT JOIN fills_shadow s ON s.fill_id = f.id
                GROUP BY f.strategy
                """
            ).fetchall()
            conn.close()
        except Exception as e:
            return web.json_response({"err": str(e)}, status=500)
        return web.json_response(
            [
                {
                    "strategy": r["strategy"],
                    "n_resolved": r["n"],
                    "realized_pnl_vwap": round(float(r["realized_pnl"] or 0), 2),
                    "realized_pnl_pessimistic": round(float(r["pess_pnl"] or 0), 2),
                    "execution_cost_estimate": round(
                        float((r["realized_pnl"] or 0) - (r["pess_pnl"] or 0)), 2
                    ),
                }
                for r in rows
            ]
        )

    async def api_health(self, request: web.Request) -> web.Response:
        # Lightweight liveness/readiness probe.
        try:
            s = self.broker.summary()
            books_quoted = sum(
                1 for b in self.book_store.books.values() if b.best_bid() and b.best_ask()
            )
        except Exception as e:
            return web.json_response({"ok": False, "err": str(e)}, status=500)
        return web.json_response(
            {
                "ok": True,
                "ts": time.time(),
                "nav": s["nav_liquidation"],
                "cash": s["cash"],
                "open_positions": s["open_positions"],
                "fills": s["fills"],
                "drawdown_pct": s["drawdown_pct"],
                "books_total": len(self.book_store.books),
                "books_quoted": books_quoted,
                "killed": s["killed"],
            }
        )

    async def index(self, request: web.Request) -> web.Response:
        return web.Response(text=DASHBOARD_HTML, content_type="text/html")

    async def run(self) -> None:
        app = web.Application()
        app.router.add_get("/", self.index)
        app.router.add_get("/api/summary", self.api_summary)
        app.router.add_get("/api/positions", self.api_positions)
        app.router.add_get("/api/fills", self.api_fills)
        app.router.add_get("/api/nav-history", self.api_nav_history)
        app.router.add_get("/api/resolutions", self.api_resolutions)
        app.router.add_get("/api/health", self.api_health)
        app.router.add_get("/api/pessimistic", self.api_pessimistic_nav)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", self.port)
        await site.start()
        log.info("dashboard_started", url=f"http://127.0.0.1:{self.port}")
        # Keep alive forever
        await asyncio.Event().wait()


DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>polyagent — paper portfolio</title>
<style>
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font: 14px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #0d1117;
    color: #e6edf3;
  }
  header {
    padding: 24px 32px 8px;
    border-bottom: 1px solid #21262d;
  }
  header h1 { margin: 0 0 4px; font-size: 18px; font-weight: 600; }
  header .subtle { color: #8b949e; font-size: 12px; }
  main { padding: 24px 32px; }
  .grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 24px;
  }
  .card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 16px;
  }
  .card .label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #8b949e;
    margin-bottom: 8px;
  }
  .card .value {
    font-size: 22px;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
  }
  .card .delta {
    margin-top: 4px;
    font-size: 12px;
    font-variant-numeric: tabular-nums;
  }
  .pos { color: #3fb950; }
  .neg { color: #f85149; }
  .neu { color: #8b949e; }
  section { margin-bottom: 32px; }
  section h2 {
    font-size: 14px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #8b949e;
    margin: 0 0 12px;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    overflow: hidden;
  }
  th, td {
    padding: 8px 12px;
    text-align: left;
    font-variant-numeric: tabular-nums;
    border-bottom: 1px solid #21262d;
  }
  th { background: #0d1117; color: #8b949e; font-weight: 500; font-size: 12px; }
  td.num { text-align: right; }
  tbody tr:last-child td { border-bottom: none; }
  tbody tr:hover { background: #1c2128; }
  .pill {
    display: inline-block;
    padding: 1px 8px;
    border-radius: 10px;
    font-size: 11px;
    font-weight: 500;
  }
  .pill-yes { background: #1f6e35; color: #b5f6c5; }
  .pill-no  { background: #6e1f1f; color: #f6b5b5; }
  .pill-cat { background: #1f3e6e; color: #b5d0f6; }
  .small { font-size: 12px; color: #8b949e; }
  .refresh {
    background: none;
    border: 1px solid #30363d;
    color: #c9d1d9;
    padding: 4px 12px;
    border-radius: 6px;
    cursor: pointer;
    font: inherit;
  }
  .refresh:hover { background: #21262d; }
  #nav-chart { width: 100%; height: 220px; }
</style>
</head>
<body>
<header>
  <h1>polyagent — paper portfolio</h1>
  <div class="subtle">
    Live paper-trading dashboard ·
    <span id="last-update">loading…</span>
    <button class="refresh" onclick="loadAll()">refresh</button>
    <span class="small" id="auto-info">auto-refreshes every 10s</span>
  </div>
</header>
<main>
  <div class="grid">
    <div class="card">
      <div class="label">total equity (liquidation)</div>
      <div class="value" id="nav">—</div>
      <div class="delta" id="nav-delta">—</div>
    </div>
    <div class="card">
      <div class="label">cash</div>
      <div class="value" id="cash">—</div>
      <div class="delta small">available for new trades</div>
    </div>
    <div class="card">
      <div class="label">positions @ bid</div>
      <div class="value" id="liq">—</div>
      <div class="delta" id="open-count">—</div>
    </div>
    <div class="card">
      <div class="label">realized P&L</div>
      <div class="value" id="realized">—</div>
      <div class="delta" id="fills-count">—</div>
    </div>
  </div>

  <section>
    <h2>NAV history (mid)</h2>
    <canvas id="nav-chart"></canvas>
  </section>

  <section>
    <h2>per-strategy P&amp;L</h2>
    <table>
      <thead>
        <tr><th>strategy</th>
          <th class="num">resolved</th>
          <th class="num">wins</th>
          <th class="num">win rate</th>
          <th class="num">realized P&amp;L</th>
          <th class="num">open</th>
          <th class="num">open notional</th>
        </tr>
      </thead>
      <tbody id="strategies"><tr><td colspan="7" class="small">loading…</td></tr></tbody>
    </table>
  </section>

  <section>
    <h2>open positions</h2>
    <table>
      <thead>
        <tr>
          <th>question</th>
          <th>cat</th>
          <th>side</th>
          <th class="num">size</th>
          <th class="num">avg cost</th>
          <th class="num">bid</th>
          <th class="num">ask</th>
          <th class="num">value@bid</th>
          <th class="num">unrealized</th>
        </tr>
      </thead>
      <tbody id="positions"><tr><td colspan="9" class="small">loading…</td></tr></tbody>
    </table>
  </section>

  <section>
    <h2>recent settlements</h2>
    <table>
      <thead><tr><th>time</th><th>question</th><th>winner</th><th class="num">P&amp;L</th></tr></thead>
      <tbody id="resolutions"><tr><td colspan="4" class="small">loading…</td></tr></tbody>
    </table>
  </section>

  <section>
    <h2>recent fills</h2>
    <table>
      <thead>
        <tr>
          <th>time</th>
          <th>strategy</th>
          <th>side</th>
          <th>question</th>
          <th class="num">price</th>
          <th class="num">size</th>
          <th class="num">notional</th>
        </tr>
      </thead>
      <tbody id="fills"><tr><td colspan="7" class="small">loading…</td></tr></tbody>
    </table>
  </section>
</main>

<script>
const fmtUsd = v => (v == null ? '—' : (v < 0 ? '-$' : '$') + Math.abs(v).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2}));
const fmtPct = v => (v == null ? '—' : (v >= 0 ? '+' : '') + v.toFixed(2) + '%');
const cls = v => v == null ? 'neu' : (v > 0 ? 'pos' : v < 0 ? 'neg' : 'neu');

async function fetchJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(r.statusText);
  return await r.json();
}

function renderSummary(data) {
  const s = data.summary;
  document.getElementById('nav').textContent = fmtUsd(s.nav_liquidation);
  const liqPct = s.pnl_pct_liquidation;
  const navDelta = document.getElementById('nav-delta');
  navDelta.textContent = fmtUsd(s.pnl_total_liquidation) + '  (' + fmtPct(liqPct) + ')';
  navDelta.className = 'delta ' + cls(liqPct);

  document.getElementById('cash').textContent = fmtUsd(s.cash);
  document.getElementById('liq').textContent = fmtUsd(s.liquidation_value);
  document.getElementById('open-count').textContent = s.open_positions + ' open positions · mid value ' + fmtUsd(s.position_value);

  const r = document.getElementById('realized');
  r.textContent = fmtUsd(s.realized_pnl);
  r.className = 'value ' + cls(s.realized_pnl);
  document.getElementById('fills-count').textContent = s.fills + ' fills · unrealized@bid ' + fmtUsd(s.unrealized_pnl_liquidation);

  const tbody = document.getElementById('strategies');
  tbody.innerHTML = '';
  if (!data.strategies.length) {
    tbody.innerHTML = '<tr><td colspan="7" class="small">no strategy data yet</td></tr>';
  } else {
    for (const s of data.strategies) {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${s.strategy}</td>
        <td class="num">${s.resolved}</td>
        <td class="num">${s.wins}</td>
        <td class="num">${s.resolved ? (100 * s.win_rate).toFixed(0) + '%' : '—'}</td>
        <td class="num ${cls(s.realized_pnl)}">${fmtUsd(s.realized_pnl)}</td>
        <td class="num">${s.open ?? 0}</td>
        <td class="num">${fmtUsd(s.open_notional ?? 0)}</td>
      `;
      tbody.appendChild(tr);
    }
  }
  const ts = new Date(data.ts * 1000).toLocaleTimeString();
  document.getElementById('last-update').textContent = 'last update ' + ts;
}

function renderPositions(rows) {
  const tbody = document.getElementById('positions');
  tbody.innerHTML = '';
  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="9" class="small">no open positions</td></tr>';
    return;
  }
  for (const p of rows) {
    const tr = document.createElement('tr');
    const sideCls = p.side === 'YES' ? 'pill-yes' : 'pill-no';
    tr.innerHTML = `
      <td>${(p.question || p.token_id).slice(0,80)}</td>
      <td>${p.category ? '<span class="pill pill-cat">'+p.category+'</span>' : '—'}</td>
      <td><span class="pill ${sideCls}">${p.side}</span></td>
      <td class="num">${p.size.toLocaleString(undefined,{maximumFractionDigits:1})}</td>
      <td class="num">${p.avg_cost.toFixed(4)}</td>
      <td class="num">${p.bid !== null ? p.bid.toFixed(4) : '—'}</td>
      <td class="num">${p.ask !== null ? p.ask.toFixed(4) : '—'}</td>
      <td class="num">${fmtUsd(p.value_bid)}</td>
      <td class="num ${cls(p.unrealized_bid)}">${fmtUsd(p.unrealized_bid)}</td>
    `;
    tbody.appendChild(tr);
  }
}

function renderFills(rows) {
  const tbody = document.getElementById('fills');
  tbody.innerHTML = '';
  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="7" class="small">no fills yet</td></tr>';
    return;
  }
  for (const f of rows) {
    const tr = document.createElement('tr');
    const yn = f.yes_no === 'YES' ? 'pill-yes' : f.yes_no === 'NO' ? 'pill-no' : 'pill-cat';
    tr.innerHTML = `
      <td class="small">${f.ts_iso}</td>
      <td>${f.strategy}</td>
      <td><span class="pill ${yn}">${f.side} ${f.yes_no}</span></td>
      <td>${(f.question || '').slice(0,80)}</td>
      <td class="num">${f.price.toFixed(4)}</td>
      <td class="num">${f.size.toLocaleString(undefined,{maximumFractionDigits:1})}</td>
      <td class="num">${fmtUsd(f.notional)}</td>
    `;
    tbody.appendChild(tr);
  }
}

function renderResolutions(rows) {
  const tbody = document.getElementById('resolutions');
  tbody.innerHTML = '';
  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="4" class="small">no resolutions yet — markets we hold will start settling soon</td></tr>';
    return;
  }
  for (const r of rows) {
    const tr = document.createElement('tr');
    const winner = r.yes_won ? '<span class="pill pill-yes">YES</span>' : '<span class="pill pill-no">NO</span>';
    tr.innerHTML = `
      <td class="small">${r.ts_iso}</td>
      <td>${(r.question || '').slice(0,80)}</td>
      <td>${winner}</td>
      <td class="num ${cls(r.pnl)}">${fmtUsd(r.pnl)}</td>
    `;
    tbody.appendChild(tr);
  }
}

function renderNavChart(rows) {
  const c = document.getElementById('nav-chart');
  const ctx = c.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  c.width = c.clientWidth * dpr; c.height = c.clientHeight * dpr;
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, c.clientWidth, c.clientHeight);
  if (!rows.length) {
    ctx.fillStyle = '#8b949e'; ctx.font='12px sans-serif';
    ctx.fillText('not enough NAV history yet', 10, 20);
    return;
  }
  const navs = rows.map(r => r.nav);
  const minN = Math.min(...navs), maxN = Math.max(...navs);
  const pad = 12;
  const W = c.clientWidth, H = c.clientHeight;
  const xScale = i => pad + (i / (rows.length - 1 || 1)) * (W - 2*pad);
  const yScale = v => pad + (1 - (v - minN) / ((maxN - minN) || 1)) * (H - 2*pad);
  // Baseline at start NAV
  const start = rows[0].nav;
  ctx.strokeStyle = '#30363d';
  ctx.lineWidth = 1;
  ctx.setLineDash([3,3]);
  ctx.beginPath();
  ctx.moveTo(pad, yScale(start));
  ctx.lineTo(W - pad, yScale(start));
  ctx.stroke();
  ctx.setLineDash([]);
  // NAV line
  ctx.strokeStyle = navs[navs.length-1] >= start ? '#3fb950' : '#f85149';
  ctx.lineWidth = 2;
  ctx.beginPath();
  rows.forEach((r, i) => {
    const x = xScale(i), y = yScale(r.nav);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
  // Range labels
  ctx.fillStyle = '#8b949e'; ctx.font='10px sans-serif';
  ctx.fillText('$' + maxN.toFixed(0), 4, 12);
  ctx.fillText('$' + minN.toFixed(0), 4, H - 4);
}

async function loadAll() {
  try {
    const [s, p, f, n, r] = await Promise.all([
      fetchJSON('/api/summary'),
      fetchJSON('/api/positions'),
      fetchJSON('/api/fills?limit=30'),
      fetchJSON('/api/nav-history?limit=500'),
      fetchJSON('/api/resolutions?limit=20'),
    ]);
    renderSummary(s);
    renderPositions(p);
    renderFills(f);
    renderNavChart(n);
    renderResolutions(r);
  } catch (e) {
    document.getElementById('last-update').textContent = 'error: ' + e.message;
  }
}
loadAll();
setInterval(loadAll, 10000);
</script>
</body>
</html>
"""
