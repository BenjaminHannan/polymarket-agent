"""Web dashboard for the paper-trading bot.

Single-process aiohttp server. Reads from the live broker for cash/positions
(no DB lock contention) and from SQLite for historical fills/resolutions/NAV.
Serves a self-contained HTML page at / and JSON endpoints under /api.

Run as a supervised task in main; default port 8080. Open
http://localhost:8080 to see:

  - sticky header with cert-gate status (ON/OFF + allowed categories)
  - headline cards: NAV, cash, positions value, realized P&L
  - NAV history chart (hover for tooltip, time axis, baseline reference)
  - strategy-certificate panel (every cert with enabled/disabled state, DSR,
    edge claim, sign-test, n_holdout)
  - by-category rollup: open positions, fills 24h/total, win rate, P&L,
    with certified categories highlighted
  - per-strategy P&L
  - open positions, recent settlements, recent fills (with category pills)

JSON endpoints:
  /api/summary        broker state + per-strategy P&L
  /api/positions      open positions with mark + unrealized
  /api/fills          recent fills (with category)
  /api/nav-history    NAV time series
  /api/resolutions    recent settlements
  /api/health         liveness probe
  /api/pessimistic    pessimistic execution-cost ledger
  /api/certificates   strategy_certificates rows + gate status
  /api/by-category    open + historical activity rolled up by category
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
            # Pull category fallback for fills against settled markets that
            # are no longer in the live scan.
            cids = {r["condition_id"] for r in rows}
            hist_cats: dict[str, str] = {}
            if cids:
                placeholders = ",".join("?" * len(cids))
                cat_rows = conn.execute(
                    f"SELECT condition_id, category FROM signal_outcomes "
                    f"WHERE category IS NOT NULL AND condition_id IN ({placeholders})",
                    list(cids),
                ).fetchall()
                hist_cats = {r["condition_id"]: r["category"] for r in cat_rows}
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
            cat = (market.category if market and market.category else
                   hist_cats.get(r["condition_id"]))
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
                    "category": cat,
                    "price": r["price"],
                    "size": r["size"],
                    "notional": r["notional"],
                }
            )
        return out

    def _certificates(self) -> dict[str, Any]:
        """Return active and superseded strategy certificates + gate status."""
        try:
            conn = _read_only_conn(self.db_path)
            rows = conn.execute(
                """SELECT name, enabled, dsr_holdout, n_holdout, issued_ts, detail
                   FROM strategy_certificates ORDER BY issued_ts DESC"""
            ).fetchall()
            conn.close()
        except Exception as e:
            log.warning("dashboard_certs_query_error", err=str(e))
            return {"certs": [], "gate_enabled": False, "allowed_categories": []}
        certs: list[dict[str, Any]] = []
        allowed: set[str] = set()
        for r in rows:
            try:
                d = json.loads(r["detail"] or "{}")
            except json.JSONDecodeError:
                d = {}
            cat = d.get("category")
            if r["enabled"] and isinstance(cat, str) and cat:
                allowed.add(cat)
            certs.append(
                {
                    "name": r["name"],
                    "enabled": bool(r["enabled"]),
                    "category": cat,
                    "dsr": r["dsr_holdout"],
                    "n_holdout": r["n_holdout"],
                    "issued_ts": r["issued_ts"],
                    "issued_iso": time.strftime("%Y-%m-%d", time.gmtime(r["issued_ts"])),
                    "reason": d.get("reason"),
                    "fold_edge_mean": d.get("fold_edge_mean"),
                    "fold_pos_count": d.get("fold_pos_count"),
                    "fold_count": d.get("fold_count"),
                    "sign_test_p_value": d.get("sign_test_p_value"),
                    "market_baseline_logloss": d.get("market_baseline_logloss"),
                    "combiner_in_sample_logloss": d.get("combiner_in_sample_logloss"),
                }
            )
        return {
            "certs": certs,
            "gate_enabled": bool(getattr(settings, "enable_certificate_gate", False)),
            "allowed_categories": sorted(allowed),
        }

    def _by_category(self) -> list[dict[str, Any]]:
        """Aggregate live positions + fills + realized P&L by market category.

        Categorisation order: live market lookup (current scan) → resolved
        market category from signal_outcomes (historical) → "uncategorized".
        """
        agg: dict[str, dict[str, float]] = {}

        def _bucket(cat: str | None) -> dict[str, float]:
            key = cat or "uncategorized"
            return agg.setdefault(
                key,
                {
                    "category": key,
                    "open_positions": 0,
                    "open_value_bid": 0.0,
                    "open_unrealized_bid": 0.0,
                    "fills_24h": 0,
                    "fills_total": 0,
                    "notional_total": 0.0,
                    "realized_pnl": 0.0,
                    "resolved": 0,
                    "wins": 0,
                },
            )

        # Build a condition_id -> category lookup from signal_outcomes for
        # historical fills whose markets are no longer in the live scan.
        try:
            conn = _read_only_conn(self.db_path)
            cat_rows = conn.execute(
                "SELECT condition_id, category FROM signal_outcomes "
                "WHERE category IS NOT NULL"
            ).fetchall()
            historical_cats = {r["condition_id"]: r["category"] for r in cat_rows}
        except Exception as e:
            log.warning("dashboard_bycat_lookup_error", err=str(e))
            historical_cats = {}

        def _resolve_cat(token_id: str, condition_id: str | None) -> str | None:
            m = self.markets_by_token.get(token_id)
            if m and m.category:
                return m.category
            if condition_id and condition_id in historical_cats:
                return historical_cats[condition_id]
            return None

        for p in self.broker.positions.values():
            if p.size <= 0:
                continue
            market = self.markets_by_token.get(p.token_id)
            cid = market.condition_id if market else None
            cat = _resolve_cat(p.token_id, cid)
            book = self.book_store.books.get(p.token_id)
            bid = book.best_bid() if book else None
            mark = bid[0] if bid else p.avg_cost
            b = _bucket(cat)
            b["open_positions"] += 1
            b["open_value_bid"] += p.size * mark
            b["open_unrealized_bid"] += (mark - p.avg_cost) * p.size

        # All-time fills + realized P&L by category
        try:
            conn = _read_only_conn(self.db_path)
            now = time.time()
            f_rows = conn.execute(
                "SELECT ts, condition_id, token_id, notional FROM fills"
            ).fetchall()
            r_rows = conn.execute(
                """SELECT f.condition_id, f.token_id, f.price, f.size,
                          r.yes_won, r.yes_token_id, r.no_token_id
                   FROM fills f INNER JOIN resolutions r
                   ON f.condition_id = r.condition_id"""
            ).fetchall()
            conn.close()
        except Exception as e:
            log.warning("dashboard_bycat_query_error", err=str(e))
            return list(agg.values())

        for r in f_rows:
            cat = _resolve_cat(r["token_id"], r["condition_id"])
            b = _bucket(cat)
            b["fills_total"] += 1
            b["notional_total"] += float(r["notional"] or 0)
            if now - r["ts"] < 86400:
                b["fills_24h"] += 1

        for r in r_rows:
            cat = _resolve_cat(r["token_id"], r["condition_id"])
            b = _bucket(cat)
            b["resolved"] += 1
            won = (r["token_id"] == r["yes_token_id"] and r["yes_won"]) or (
                r["token_id"] == r["no_token_id"] and not r["yes_won"]
            )
            if won:
                b["wins"] += 1
            payout = 1.0 if won else 0.0
            b["realized_pnl"] += (payout - r["price"]) * r["size"]

        out = list(agg.values())
        for b in out:
            b["open_value_bid"] = round(b["open_value_bid"], 2)
            b["open_unrealized_bid"] = round(b["open_unrealized_bid"], 2)
            b["notional_total"] = round(b["notional_total"], 2)
            b["realized_pnl"] = round(b["realized_pnl"], 2)
            b["win_rate"] = (b["wins"] / b["resolved"]) if b["resolved"] else None
        out.sort(
            key=lambda x: (
                -(x["open_value_bid"] or 0) - abs(x["realized_pnl"] or 0) * 0.5
            )
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

    async def api_certificates(self, request: web.Request) -> web.Response:
        return web.json_response(self._certificates())

    async def api_by_category(self, request: web.Request) -> web.Response:
        return web.json_response(self._by_category())

    async def api_failures(self, request: web.Request) -> web.Response:
        return web.json_response(self._failures())

    def _failures(self) -> dict:
        """Aggregate model_failures: counts by type, by category, recent rows."""
        try:
            conn = _read_only_conn(self.db_path)
            by_type = [
                dict(r) for r in conn.execute(
                    """SELECT failure_type, COUNT(*) AS n,
                              ROUND(AVG(severity), 4) AS avg_severity,
                              ROUND(AVG(log_loss_model), 4) AS avg_log_loss
                       FROM model_failures
                       GROUP BY failure_type ORDER BY 2 DESC"""
                )
            ]
            by_cat = [
                dict(r) for r in conn.execute(
                    """SELECT COALESCE(category, '(none)') AS category,
                              COUNT(*) AS n,
                              ROUND(AVG(severity), 4) AS avg_severity
                       FROM model_failures
                       GROUP BY category ORDER BY 2 DESC LIMIT 12"""
                )
            ]
            recent = [
                dict(r) for r in conn.execute(
                    """SELECT condition_id, resolved_ts, yes_won,
                              p_model, p_market, failure_type, severity,
                              category, question, notional_traded, realized_pnl
                       FROM model_failures
                       ORDER BY resolved_ts DESC LIMIT 25"""
                )
            ]
            worst = [
                dict(r) for r in conn.execute(
                    """SELECT severity, p_model, p_market, yes_won,
                              failure_type, category, question
                       FROM model_failures
                       WHERE failure_type IN
                             ('high_confidence_wrong','model_loud_wrong_market_right')
                       ORDER BY severity DESC LIMIT 10"""
                )
            ]
            total = conn.execute("SELECT COUNT(*) FROM model_failures").fetchone()[0]
            conn.close()
        except Exception as e:
            log.warning("dashboard_failures_query_error", err=str(e))
            return {"total": 0, "by_type": [], "by_category": [], "recent": [], "worst": []}
        # ISO-format timestamps for display
        for r in recent:
            r["ts_iso"] = time.strftime("%Y-%m-%d", time.gmtime(r.get("resolved_ts") or 0))
        return {"total": int(total), "by_type": by_type, "by_category": by_cat,
                "recent": recent, "worst": worst}

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
        app.router.add_get("/api/certificates", self.api_certificates)
        app.router.add_get("/api/by-category", self.api_by_category)
        app.router.add_get("/api/failures", self.api_failures)
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
  :root {
    --bg: #0d1117;
    --panel: #161b22;
    --panel-2: #1c2128;
    --border: #21262d;
    --border-2: #30363d;
    --text: #e6edf3;
    --muted: #8b949e;
    --accent: #58a6ff;
    --pos: #3fb950;
    --neg: #f85149;
    --warn: #d29922;
  }
  body {
    margin: 0;
    font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
  }
  header {
    position: sticky; top: 0; z-index: 10;
    padding: 14px 32px;
    background: rgba(13,17,23,0.85);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
  }
  header h1 {
    margin: 0; font-size: 16px; font-weight: 600;
    display: flex; align-items: center; gap: 8px;
  }
  header h1 .dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--pos); box-shadow: 0 0 8px var(--pos);
    animation: pulse 2s infinite ease-in-out;
  }
  @keyframes pulse { 50% { opacity: 0.4; } }
  header .pill-gate {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 11px; padding: 3px 10px; border-radius: 12px;
    background: #0d2f1d; color: #7ee2a8; border: 1px solid #1f6e35;
  }
  header .pill-gate.off {
    background: #2d2018; color: #f0a86a; border-color: #6e4a1f;
  }
  header .spacer { flex: 1; }
  header .last-update { color: var(--muted); font-size: 12px; }
  header button.refresh {
    background: none; border: 1px solid var(--border-2);
    color: var(--text); padding: 5px 12px; border-radius: 6px;
    cursor: pointer; font: inherit;
  }
  header button.refresh:hover { background: var(--panel-2); }
  main { padding: 24px 32px; max-width: 1600px; margin: 0 auto; }
  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 14px;
    margin-bottom: 24px;
  }
  .card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 18px;
    transition: border-color 0.15s;
  }
  .card:hover { border-color: var(--border-2); }
  .card .label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--muted);
    margin-bottom: 6px;
    display: flex; align-items: center; justify-content: space-between;
  }
  .card .value {
    font-size: 26px;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
    line-height: 1.1;
  }
  .card .delta {
    margin-top: 6px;
    font-size: 12px;
    font-variant-numeric: tabular-nums;
  }
  .pos { color: var(--pos); }
  .neg { color: var(--neg); }
  .neu { color: var(--muted); }
  .warn { color: var(--warn); }
  section { margin-bottom: 28px; }
  section h2 {
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted);
    margin: 0 0 10px;
    display: flex; align-items: center; gap: 8px;
  }
  section h2 .count {
    font-size: 11px; color: var(--muted);
    background: var(--panel); padding: 1px 8px; border-radius: 10px;
    border: 1px solid var(--border); text-transform: none;
    letter-spacing: 0;
  }
  table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
  }
  th, td {
    padding: 9px 14px;
    text-align: left;
    font-variant-numeric: tabular-nums;
    border-bottom: 1px solid var(--border);
  }
  th {
    background: #0a0e14;
    color: var(--muted);
    font-weight: 500;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    /* No sticky positioning: each table is short enough that pinning
       headers to the viewport caused them to slide into adjacent rows
       (specifically appearing between data rows in by-category). */
  }
  td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
  tbody tr:last-child td { border-bottom: none; }
  tbody tr:hover { background: var(--panel-2); }
  tr.certified { background: rgba(63,185,80,0.04); }
  .pill {
    display: inline-block;
    padding: 2px 9px;
    border-radius: 11px;
    font-size: 11px;
    font-weight: 500;
    line-height: 1.4;
  }
  .pill-yes { background: #133021; color: #7ee2a8; }
  .pill-no  { background: #301414; color: #f6b5b5; }
  .pill-cat { background: #1f2d44; color: #9bbcdc; }
  .pill-cat.allowed { background: #133021; color: #7ee2a8; }
  .pill-on  { background: #133021; color: #7ee2a8; }
  .pill-off { background: #2d2018; color: #f0a86a; }
  .small { font-size: 12px; color: var(--muted); }
  .mono { font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; }
  .cert-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    gap: 12px;
  }
  .cert-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 16px;
    border-left: 3px solid var(--border-2);
  }
  .cert-card.enabled { border-left-color: var(--pos); }
  .cert-card.disabled { border-left-color: var(--neg); }
  .cert-card.disabled .name { color: #8b949e; }
  .cert-card .name {
    font-weight: 600; font-size: 13px;
    display: flex; justify-content: space-between; align-items: center;
    gap: 8px;
    word-break: break-word;
  }
  .cert-card .row {
    margin-top: 6px; display: flex; justify-content: space-between;
    font-size: 12px; color: #9ca3a9;
  }
  .cert-card .row .v {
    color: var(--text); font-variant-numeric: tabular-nums;
    font-weight: 500;
  }
  .cert-card .reason {
    font-size: 11px; color: #9ca3a9; margin-top: 10px; line-height: 1.45;
    padding-top: 10px; border-top: 1px solid var(--border);
  }
  #nav-chart-wrap {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px;
  }
  #nav-chart { width: 100%; height: 280px; display: block; }
  .failure-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
    gap: 14px;
  }
  #nav-tip {
    position: absolute;
    pointer-events: none;
    background: var(--panel-2);
    border: 1px solid var(--border-2);
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 12px;
    font-variant-numeric: tabular-nums;
    transform: translate(-50%, -120%);
    display: none;
    white-space: nowrap;
  }
</style>
</head>
<body>
<header>
  <h1><span class="dot"></span>polyagent <span class="small" style="font-weight:400">paper</span></h1>
  <span class="pill-gate" id="gate-pill">cert gate: …</span>
  <span class="spacer"></span>
  <span class="last-update" id="last-update">loading…</span>
  <button class="refresh" onclick="loadAll()">refresh</button>
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
    <h2>NAV history (mid) <span class="small" id="nav-range">—</span></h2>
    <div id="nav-chart-wrap" style="position:relative">
      <canvas id="nav-chart"></canvas>
      <div id="nav-tip"></div>
    </div>
  </section>

  <section>
    <h2>strategy certificates <span class="count" id="cert-count">0</span></h2>
    <div class="cert-grid" id="certificates"><div class="small">loading…</div></div>
  </section>

  <section>
    <h2>by category</h2>
    <table>
      <thead>
        <tr>
          <th>category</th>
          <th>cert</th>
          <th class="num">open</th>
          <th class="num">value@bid</th>
          <th class="num">unrealized</th>
          <th class="num">fills (24h / total)</th>
          <th class="num">notional</th>
          <th class="num">resolved</th>
          <th class="num">win rate</th>
          <th class="num">realized P&amp;L</th>
        </tr>
      </thead>
      <tbody id="by-category"><tr><td colspan="10" class="small">loading…</td></tr></tbody>
    </table>
  </section>

  <section>
    <h2>model failures <span class="count" id="failures-count">0</span></h2>
    <div class="failure-grid">
      <div>
        <h3 class="small" style="color:var(--muted);text-transform:uppercase;letter-spacing:0.05em;margin:0 0 8px 0">by type</h3>
        <table>
          <thead><tr><th>failure type</th><th class="num">n</th><th class="num">avg severity</th><th class="num">avg log-loss</th></tr></thead>
          <tbody id="failures-by-type"><tr><td colspan="4" class="small">loading…</td></tr></tbody>
        </table>
      </div>
      <div>
        <h3 class="small" style="color:var(--muted);text-transform:uppercase;letter-spacing:0.05em;margin:0 0 8px 0">by category</h3>
        <table>
          <thead><tr><th>category</th><th class="num">n</th><th class="num">avg severity</th></tr></thead>
          <tbody id="failures-by-category"><tr><td colspan="3" class="small">loading…</td></tr></tbody>
        </table>
      </div>
    </div>
    <h3 class="small" style="color:var(--muted);text-transform:uppercase;letter-spacing:0.05em;margin:18px 0 8px 0">worst high-confidence misses</h3>
    <table>
      <thead><tr><th>severity</th><th>type</th><th>cat</th><th class="num">p_model</th><th class="num">p_mkt</th><th>actual</th><th>question</th></tr></thead>
      <tbody id="failures-worst"><tr><td colspan="7" class="small">loading…</td></tr></tbody>
    </table>
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
    <h2>recent fills <span class="count" id="fills-count-badge">0</span></h2>
    <table>
      <thead>
        <tr>
          <th>time</th>
          <th>strategy</th>
          <th>cat</th>
          <th>side</th>
          <th>question</th>
          <th class="num">price</th>
          <th class="num">size</th>
          <th class="num">notional</th>
        </tr>
      </thead>
      <tbody id="fills"><tr><td colspan="8" class="small">loading…</td></tr></tbody>
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

let _allowedCategories = new Set();

function renderFills(rows) {
  const tbody = document.getElementById('fills');
  document.getElementById('fills-count-badge').textContent = rows.length;
  tbody.innerHTML = '';
  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="8" class="small">no fills yet</td></tr>';
    return;
  }
  for (const f of rows) {
    const tr = document.createElement('tr');
    const yn = f.yes_no === 'YES' ? 'pill-yes' : f.yes_no === 'NO' ? 'pill-no' : 'pill-cat';
    const catCls = f.category && _allowedCategories.has(f.category) ? 'pill pill-cat allowed' : 'pill pill-cat';
    const catCell = f.category ? `<span class="${catCls}">${f.category}</span>` : '<span class="small">—</span>';
    tr.innerHTML = `
      <td class="small mono">${f.ts_iso.slice(11)}</td>
      <td>${f.strategy}</td>
      <td>${catCell}</td>
      <td><span class="pill ${yn}">${f.side} ${f.yes_no}</span></td>
      <td>${(f.question || '').slice(0,90)}</td>
      <td class="num">${f.price.toFixed(4)}</td>
      <td class="num">${f.size.toLocaleString(undefined,{maximumFractionDigits:1})}</td>
      <td class="num">${fmtUsd(f.notional)}</td>
    `;
    tbody.appendChild(tr);
  }
}

function renderCertificates(data) {
  const grid = document.getElementById('certificates');
  const certs = data.certs || [];
  document.getElementById('cert-count').textContent = certs.length + ' total · ' +
    certs.filter(c => c.enabled).length + ' enabled';

  // Header gate pill
  const gatePill = document.getElementById('gate-pill');
  if (data.gate_enabled) {
    gatePill.textContent = 'cert gate ON · ' + (data.allowed_categories.join(', ') || 'no categories');
    gatePill.className = 'pill-gate';
  } else {
    gatePill.textContent = 'cert gate OFF';
    gatePill.className = 'pill-gate off';
  }
  _allowedCategories = new Set(data.allowed_categories || []);

  grid.innerHTML = '';
  if (!certs.length) {
    grid.innerHTML = '<div class="small">no certificates issued yet</div>';
    return;
  }
  for (const c of certs) {
    const div = document.createElement('div');
    div.className = 'cert-card ' + (c.enabled ? 'enabled' : 'disabled');
    const edgeStr = c.fold_edge_mean != null ?
      (c.fold_edge_mean >= 0 ? '+' : '') + c.fold_edge_mean.toFixed(4) : '—';
    const dsrStr = c.dsr != null ? c.dsr.toFixed(4) : '—';
    const foldStr = (c.fold_pos_count != null && c.fold_count != null) ?
      `${c.fold_pos_count}/${c.fold_count} folds positive` : '—';
    div.innerHTML = `
      <div class="name">
        <span>${c.name}</span>
        <span class="pill ${c.enabled ? 'pill-on' : 'pill-off'}">${c.enabled ? 'ENABLED' : 'disabled'}</span>
      </div>
      <div class="row"><span>category</span><span class="v">${c.category || '—'}</span></div>
      <div class="row"><span>n holdout</span><span class="v">${(c.n_holdout || 0).toLocaleString()}</span></div>
      <div class="row"><span>DSR</span><span class="v">${dsrStr}</span></div>
      <div class="row"><span>edge (log-loss)</span><span class="v ${cls(c.fold_edge_mean)}">${edgeStr}</span></div>
      <div class="row"><span>folds positive</span><span class="v">${foldStr}</span></div>
      <div class="row"><span>issued</span><span class="v">${c.issued_iso}</span></div>
      <div class="reason">${c.reason || ''}</div>
    `;
    grid.appendChild(div);
  }
}

function renderFailures(data) {
  document.getElementById('failures-count').textContent =
    (data.total || 0) + ' total · ' +
    (data.by_type || []).filter(r => r.failure_type === 'high_confidence_wrong')
      .reduce((s, r) => s + (r.n || 0), 0) + ' high-confidence';

  const tBody = document.getElementById('failures-by-type');
  tBody.innerHTML = '';
  if (!data.by_type || !data.by_type.length) {
    tBody.innerHTML = '<tr><td colspan="4" class="small">no failures recorded yet</td></tr>';
  } else {
    for (const r of data.by_type) {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${r.failure_type}</td>
        <td class="num">${r.n}</td>
        <td class="num">${r.avg_severity?.toFixed?.(3) ?? '—'}</td>
        <td class="num">${r.avg_log_loss?.toFixed?.(3) ?? '—'}</td>
      `;
      tBody.appendChild(tr);
    }
  }

  const cBody = document.getElementById('failures-by-category');
  cBody.innerHTML = '';
  if (!data.by_category || !data.by_category.length) {
    cBody.innerHTML = '<tr><td colspan="3" class="small">—</td></tr>';
  } else {
    for (const r of data.by_category) {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><span class="pill pill-cat">${r.category}</span></td>
        <td class="num">${r.n}</td>
        <td class="num">${r.avg_severity?.toFixed?.(3) ?? '—'}</td>
      `;
      cBody.appendChild(tr);
    }
  }

  const wBody = document.getElementById('failures-worst');
  wBody.innerHTML = '';
  if (!data.worst || !data.worst.length) {
    wBody.innerHTML = '<tr><td colspan="7" class="small">—</td></tr>';
  } else {
    for (const r of data.worst) {
      const tr = document.createElement('tr');
      const actualPill = r.yes_won ? '<span class="pill pill-yes">YES</span>' : '<span class="pill pill-no">NO</span>';
      const ftShort = r.failure_type.replace('_', ' ').replace(/_/g, ' ');
      tr.innerHTML = `
        <td class="num">${(r.severity ?? 0).toFixed(3)}</td>
        <td class="small">${ftShort}</td>
        <td>${r.category ? `<span class="pill pill-cat">${r.category}</span>` : '<span class="small">—</span>'}</td>
        <td class="num">${(r.p_model ?? 0).toFixed(3)}</td>
        <td class="num">${r.p_market !== null && r.p_market !== undefined ? r.p_market.toFixed(3) : '—'}</td>
        <td>${actualPill}</td>
        <td>${(r.question || '').slice(0,90)}</td>
      `;
      wBody.appendChild(tr);
    }
  }
}

function renderByCategory(rows) {
  const tbody = document.getElementById('by-category');
  tbody.innerHTML = '';
  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="10" class="small">no activity yet</td></tr>';
    return;
  }
  for (const b of rows) {
    const isCert = _allowedCategories.has(b.category);
    const tr = document.createElement('tr');
    if (isCert) tr.classList.add('certified');
    const certPill = isCert ? '<span class="pill pill-on">cert</span>' : '<span class="small">—</span>';
    const winStr = b.win_rate != null ? (100 * b.win_rate).toFixed(0) + '%' : '—';
    tr.innerHTML = `
      <td><span class="pill pill-cat ${isCert ? 'allowed' : ''}">${b.category}</span></td>
      <td>${certPill}</td>
      <td class="num">${b.open_positions}</td>
      <td class="num">${fmtUsd(b.open_value_bid)}</td>
      <td class="num ${cls(b.open_unrealized_bid)}">${fmtUsd(b.open_unrealized_bid)}</td>
      <td class="num small">${b.fills_24h} / ${b.fills_total}</td>
      <td class="num">${fmtUsd(b.notional_total)}</td>
      <td class="num">${b.resolved}</td>
      <td class="num">${winStr}</td>
      <td class="num ${cls(b.realized_pnl)}">${fmtUsd(b.realized_pnl)}</td>
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

let _navRows = [];

function renderNavChart(rows) {
  _navRows = rows;
  const c = document.getElementById('nav-chart');
  const ctx = c.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  c.width = c.clientWidth * dpr; c.height = c.clientHeight * dpr;
  ctx.scale(dpr, dpr);
  const W = c.clientWidth, H = c.clientHeight;
  ctx.clearRect(0, 0, W, H);
  document.getElementById('nav-tip').style.display = 'none';
  const rangeLabel = document.getElementById('nav-range');

  if (!rows.length) {
    ctx.fillStyle = '#8b949e'; ctx.font='12px sans-serif';
    ctx.fillText('not enough NAV history yet', 16, 24);
    rangeLabel.textContent = '';
    return;
  }
  const navs = rows.map(r => r.nav);
  const minN = Math.min(...navs), maxN = Math.max(...navs);
  const span = maxN - minN || 1;
  const pad = { l: 56, r: 12, t: 14, b: 24 };
  const xScale = i => pad.l + (i / (rows.length - 1 || 1)) * (W - pad.l - pad.r);
  const yScale = v => pad.t + (1 - (v - minN) / span) * (H - pad.t - pad.b);

  // Y-axis grid + labels (5 ticks)
  ctx.strokeStyle = '#21262d';
  ctx.fillStyle = '#8b949e';
  ctx.font = '10px ui-monospace, monospace';
  ctx.lineWidth = 1;
  for (let k = 0; k <= 4; k++) {
    const v = minN + (span * k / 4);
    const y = yScale(v);
    ctx.beginPath();
    ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y);
    ctx.stroke();
    ctx.fillText('$' + v.toFixed(0), 4, y + 3);
  }

  // X-axis time labels (start, mid, end)
  const fmtTime = ts => {
    const d = new Date(ts * 1000);
    return d.getHours().toString().padStart(2,'0') + ':' +
           d.getMinutes().toString().padStart(2,'0');
  };
  ctx.textAlign = 'center';
  for (const k of [0, 0.5, 1]) {
    const i = Math.round(k * (rows.length - 1));
    ctx.fillText(fmtTime(rows[i].ts), xScale(i), H - 8);
  }
  ctx.textAlign = 'start';

  // Baseline at start NAV (dashed)
  const start = rows[0].nav;
  ctx.strokeStyle = '#30363d';
  ctx.setLineDash([4,4]);
  ctx.beginPath();
  ctx.moveTo(pad.l, yScale(start));
  ctx.lineTo(W - pad.r, yScale(start));
  ctx.stroke();
  ctx.setLineDash([]);

  // Filled area under curve
  const last = navs[navs.length-1];
  const lineColor = last >= start ? '#3fb950' : '#f85149';
  const fillColor = last >= start ? 'rgba(63,185,80,0.10)' : 'rgba(248,81,73,0.10)';
  ctx.fillStyle = fillColor;
  ctx.beginPath();
  rows.forEach((r, i) => {
    const x = xScale(i), y = yScale(r.nav);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.lineTo(xScale(rows.length-1), H - pad.b);
  ctx.lineTo(xScale(0), H - pad.b);
  ctx.closePath();
  ctx.fill();

  // NAV line
  ctx.strokeStyle = lineColor;
  ctx.lineWidth = 2;
  ctx.beginPath();
  rows.forEach((r, i) => {
    const x = xScale(i), y = yScale(r.nav);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Latest dot
  const lx = xScale(rows.length - 1), ly = yScale(last);
  ctx.fillStyle = lineColor;
  ctx.beginPath(); ctx.arc(lx, ly, 4, 0, 2*Math.PI); ctx.fill();
  ctx.strokeStyle = lineColor + '40'; // halo
  ctx.lineWidth = 6;
  ctx.beginPath(); ctx.arc(lx, ly, 8, 0, 2*Math.PI); ctx.stroke();

  rangeLabel.textContent =
    `$${minN.toFixed(0)} – $${maxN.toFixed(0)} · ` +
    `${rows.length} pts · ` +
    `${((last - start) / start * 100).toFixed(2)}% from start`;

  // Hover tooltip
  c.onmousemove = e => {
    const rect = c.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    if (mx < pad.l || mx > W - pad.r) {
      document.getElementById('nav-tip').style.display = 'none';
      return;
    }
    const frac = (mx - pad.l) / (W - pad.l - pad.r);
    const idx = Math.max(0, Math.min(rows.length - 1, Math.round(frac * (rows.length - 1))));
    const r = rows[idx];
    const tip = document.getElementById('nav-tip');
    tip.style.display = 'block';
    const wrap = document.getElementById('nav-chart-wrap');
    const wrapRect = wrap.getBoundingClientRect();
    tip.style.left = (e.clientX - wrapRect.left) + 'px';
    tip.style.top = (yScale(r.nav) + wrap.getBoundingClientRect().top - wrapRect.top + 12) + 'px';
    const dt = new Date(r.ts * 1000);
    tip.innerHTML = `<b>${fmtUsd(r.nav)}</b> &nbsp;<span class="small">${dt.toLocaleTimeString()}</span>`;
  };
  c.onmouseleave = () => { document.getElementById('nav-tip').style.display = 'none'; };
}

async function loadAll() {
  try {
    // Certificates first so _allowedCategories is set before fills/positions
    // render and pick up the certified-category highlighting.
    const certs = await fetchJSON('/api/certificates');
    renderCertificates(certs);

    const [s, p, f, n, r, bc, fl] = await Promise.all([
      fetchJSON('/api/summary'),
      fetchJSON('/api/positions'),
      fetchJSON('/api/fills?limit=40'),
      fetchJSON('/api/nav-history?limit=500'),
      fetchJSON('/api/resolutions?limit=20'),
      fetchJSON('/api/by-category'),
      fetchJSON('/api/failures'),
    ]);
    renderSummary(s);
    renderByCategory(bc);
    renderFailures(fl);
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
window.addEventListener('resize', () => renderNavChart(_navRows));
</script>
</body>
</html>
"""
