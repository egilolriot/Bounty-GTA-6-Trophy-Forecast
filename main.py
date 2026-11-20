"""gta6-trophy-rarity-forecast-01 entry (Python 3.13, standard library only).

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, five
quantiles of its earn rate on Exophase at the case's target instant.

Model: per-trophy log-time logit trend with tier shrinkage and an
uncertainty-aware spread. See METHOD.md for the full reasoning.

In short, for each trophy:
  - take its earn-rate series from the case history, in logit space
    (rates clamped to 0.01..99.99 before the logit, as the scorer does);
  - fit logit(rate) = a + b * ln(1 + hours since release) by recency-weighted
    least squares over the last 48 hours of reads (or all reads when fewer),
    so the fit follows the regime the game is in now, not launch night;
  - shrink the slope toward its tier's mean slope (bronze / silver / gold /
    platinum trophies progress at different speeds), with the shrinkage
    weight set by the fit's own standard error;
  - anchor the median at the last read's rate and move it by the shrunk
    slope over the horizon, with the total logit move capped at +/-3;
  - widen the spread with the horizon (random-walk term, as the house
    baseline does), the slope's standard error times the extrapolation
    distance, the fit's residual volatility, and the size of the jumps
    Exophase's refreshes make between consecutive distinct reads.

The program reads every <id>.in in the input directory and writes
<id>.out to the output directory:
    {"trophies": [{"id": 1, "q": [q05, q25, q50, q75, q95]}, ...]}
one row per trophy of the case, quantiles non-decreasing, each in
[0.01, 99.99]. No network, no randomness, no third-party packages: two
runs on the same case write the same bytes.

Run locally:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file would import. (This entry is one file.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
NORMAL = NormalDist()

# Random-walk spread of a rate's move, in logit units per square-root day
# (the house baseline's sigma).
SIGMA_RW = 0.5
# Floor and cap on the total logit spread, so a flat history never claims
# false precision and a wild fit never claims the rate could be anything.
SPREAD_FLOOR = 0.30
SPREAD_CAP = 2.5
# Cap on the median's total logit move away from the last read rate.
MAX_LOGIT_MOVE = 3.0
# Cap on a shrunk slope (logit per unit of ln(1 + hours)).
SLOPE_CAP = 3.0
# Fit window and recency half-life, in hours.
FIT_WINDOW_HOURS = 48.0
HALF_LIFE_HOURS = 24.0
# Tier fallback medians (percent) when a trophy has no usable rate at all.
TIER_FALLBACK = {"platinum": 0.3, "gold": 0.5, "silver": 0.7, "bronze": 1.0}


def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return clamp_pct(100.0 / (1.0 + math.exp(-lg)))
    e = math.exp(lg)
    return clamp_pct(100.0 * e / (1.0 + e))


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up, as the starter rounds it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def release_time(case: dict) -> datetime:
    """Midnight UTC of the game's release date, else the first read."""
    game = case.get("game") or {}
    raw = game.get("release_date")
    if isinstance(raw, str) and raw:
        try:
            dt = parse_time(raw)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
        try:
            d = datetime.strptime(raw[:10], "%Y-%m-%d")
            return d.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    history = case.get("history") or []
    if history:
        return parse_time(history[0]["at"])
    return datetime(2026, 11, 19, tzinfo=timezone.utc)


def wls(xs, ys, ws):
    """Weighted least squares y = a + b*x. Returns (a, b, se_b, resid_sd).

    se_b is None when the slope is not identified (one distinct x, or a
    perfect fit with no residual and fewer than 3 points to judge it by).
    """
    sw = sum(ws)
    if sw <= 0 or len(xs) < 2:
        return None
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for _, y, w in zip(xs, ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return None
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    n = len(xs)
    if n >= 3:
        # Weighted residual variance, then the slope's standard error.
        s2 = sum(w * r * r for r, w in zip(resid, ws)) / max(sw - 2.0 * sw / n, 1e-12)
        se_b = math.sqrt(max(s2, 0.0) / sxx) if s2 > 0 else 0.0
        resid_sd = math.sqrt(max(s2, 0.0))
    else:
        se_b = None
        resid_sd = 0.0
    return a, b, se_b, resid_sd


def median_jump(history, trophies):
    """Median |logit jump| between consecutive distinct reads, over all
    trophies: the scale of the steps Exophase's refreshes make. None when
    no read ever differs from the one before it."""
    jumps = []
    for i in range(len(trophies)):
        prev = None
        for read in history:
            rates = read.get("rates") or []
            if i >= len(rates) or rates[i] is None:
                continue
            cur = logit(rates[i])
            if prev is not None and abs(cur - prev) > 1e-9:
                jumps.append(abs(cur - prev))
            prev = cur
    if not jumps:
        return None
    jumps.sort()
    mid = len(jumps) // 2
    return jumps[mid] if len(jumps) % 2 else 0.5 * (jumps[mid - 1] + jumps[mid])


def forecast_case(case: dict, name: str) -> dict:
    levels = case.get("levels") or LEVELS
    z = [NORMAL.inv_cdf(tau) for tau in levels]
    trophies = case["trophies"]
    history = sorted(case.get("history") or [], key=lambda r: r["at"])
    if not history:
        # No reads at all: tier fallbacks with a wide spread.
        rows = []
        for trophy in trophies:
            med = TIER_FALLBACK.get(trophy.get("tier"), 1.0)
            rows.append({"id": trophy["id"], "q": quantiles(med, 1.5, z)})
        return {"trophies": rows}

    t0 = release_time(case)
    checkpoint = parse_time(case["at"])
    target = parse_time(case["target"])
    horizon_days = max(case.get("horizon_minutes", 0) / 1440.0, 0.0)
    x_target = math.log1p(max((target - t0).total_seconds() / 3600.0, 0.0))
    x_last = math.log1p(max((checkpoint - t0).total_seconds() / 3600.0, 0.0))

    last = history[-1]
    last_rates = last.get("rates") or []
    jump = median_jump(history, trophies)
    jump_term = 0.25 if jump is None else jump

    # Fit window: the last FIT_WINDOW_HOURS of reads when there are enough,
    # else the whole history. Recency weights halve every HALF_LIFE_HOURS.
    t_last = parse_time(last["at"])
    window = [r for r in history
              if (t_last - parse_time(r["at"])).total_seconds() / 3600.0 <= FIT_WINDOW_HOURS]
    fit_reads = window if len(window) >= 6 else history

    # Per-trophy fits, then tier-mean slopes for shrinkage.
    fits = []
    for i, trophy in enumerate(trophies):
        xs, ys, ws = [], [], []
        for read in fit_reads:
            rates = read.get("rates") or []
            if i >= len(rates) or rates[i] is None:
                continue
            t = parse_time(read["at"])
            hours = max((t - t0).total_seconds() / 3600.0, 0.0)
            xs.append(math.log1p(hours))
            ys.append(logit(rates[i]))
            age = max((t_last - t).total_seconds() / 3600.0, 0.0)
            ws.append(0.5 ** (age / HALF_LIFE_HOURS))
        fits.append(wls(xs, ys, ws) if len(xs) >= 2 else None)

    tier_slopes = {}
    for tier in ("platinum", "gold", "silver", "bronze"):
        num, den = 0.0, 0.0
        for trophy, fit in zip(trophies, fits):
            if trophy.get("tier") != tier or fit is None:
                continue
            _, b, se_b, _ = fit
            w = 1.0 / max(se_b if se_b is not None else 0.2, 0.05) ** 2
            w = min(w, 400.0)
            num += w * b
            den += w
        if den > 0:
            tier_slopes[tier] = num / den

    rows = []
    for i, trophy in enumerate(trophies):
        try:
            rows.append({"id": trophy["id"], "q": forecast_trophy(
                trophy, fits[i], tier_slopes.get(trophy.get("tier")),
                last_rates[i] if i < len(last_rates) else None,
                x_target, x_last, horizon_days, jump_term, z)})
        except Exception:
            med = TIER_FALLBACK.get(trophy.get("tier"), 1.0)
            rows.append({"id": trophy["id"], "q": quantiles(med, 1.5, z)})
    return {"trophies": rows}


def forecast_trophy(trophy, fit, tier_slope, last_rate, x_target, x_last,
                   horizon_days, jump_term, z):
    """One trophy's five quantiles."""
    tier = trophy.get("tier")
    if last_rate is None:
        med = TIER_FALLBACK.get(tier, 1.0)
        return quantiles(med, 1.5, z)
    last_logit = logit(last_rate)
    dx = x_target - x_last

    if fit is None:
        slope = tier_slope if tier_slope is not None else 0.0
        se_slope = 0.5
        resid = 0.0
    else:
        _, b, se_b, resid = fit
        if se_b is None:
            # Slope identified but its error is not: trust the fit weakly.
            slope = b
            se_slope = 0.5
        else:
            # Shrink toward the tier mean, weighted by the fit's precision.
            w_fit = min(1.0 / max(se_b, 0.05) ** 2, 400.0)
            w_prior = 25.0
            prior = tier_slope if tier_slope is not None else 0.0
            slope = (w_fit * b + w_prior * prior) / (w_fit + w_prior)
            se_slope = 1.0 / math.sqrt(w_fit + w_prior)
        slope = max(-SLOPE_CAP, min(SLOPE_CAP, slope))

    move = max(-MAX_LOGIT_MOVE, min(MAX_LOGIT_MOVE, slope * dx))
    med_logit = last_logit + move

    spread2 = (SIGMA_RW * math.sqrt(horizon_days)) ** 2
    spread2 += (se_slope * abs(dx)) ** 2
    spread2 += resid ** 2
    spread2 += 0.25 * jump_term ** 2
    if last_rate <= 0.0 or last_rate >= 100.0:
        # A censored rate (0.00% or 100.00% as printed) hides the truth.
        spread2 += 0.3 ** 2
    spread = min(SPREAD_CAP, max(SPREAD_FLOOR, math.sqrt(spread2)))

    qs = []
    prev = None
    for zi in z:
        q = round4(inv_logit(med_logit + zi * spread))
        q = float(q)
        if prev is not None and q < prev:
            q = prev
        qs.append(q)
        prev = q
    return qs


def quantiles(median_pct: float, spread: float, z) -> list:
    """Quantiles around a median given as a percent, for fallbacks."""
    med_logit = logit(median_pct)
    qs, prev = [], None
    for zi in z:
        q = float(round4(inv_logit(med_logit + zi * spread)))
        if prev is not None and q < prev:
            q = prev
        qs.append(q)
        prev = q
    return qs


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
