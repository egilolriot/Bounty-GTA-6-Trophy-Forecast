# Method — GTA VI trophy rarity forecast (week one)

Entry for make.fun `gta6-trophy-rarity-forecast-01`: forecast, for every
PlayStation 5 trophy of Grand Theft Auto VI, its earn rate on Exophase at
2026-11-26T00:00:00Z, as five quantiles (5 / 25 / 50 / 75 / 95%).

## What is being forecast

Exophase's trophy page prints, per trophy, the percent of the players it
tracks for the game who have earned that trophy. The program never sees the
future: it runs offline on the case file alone (`history`: every valid read
of the page before the checkpoint) and must project each trophy's rate
forward to the target — 6,510 minutes (4.52 days) past the deadline for the
final case.

## Why a trend model, not persistence

The house baseline says every rate stays where the last read put it. That
is the right answer for a 90-minute horizon with a flat history, and it is
why the baseline scores ~1.51 on the practice cases (whose histories are
perfectly flat launch-night reads, then one refresh). But over 4.5 days in
launch week, rates move a lot:

- **Players keep arriving.** Every new tracked player who has not earned a
  trophy yet dilutes its rate — the first refresh on launch night dropped
  nearly every rate while tracked players rose 638 → 3,868.
- **Players keep playing.** The longer the window, the more of the tracked
  players finish missions, side content and collectibles, pushing early and
  mid-game trophy rates up — often several-fold over a week.
- The net direction differs per trophy: story trophies early in the game
  rise fast; late-game, grind and platinum trophies barely move.

So the model fits a trend per trophy instead of assuming persistence, and
lets the data decide the direction.

## The model (`main.py`)

For each trophy, work in logit space (the scorer's scale: a miss from 1%
to 2% costs about what a miss from 33% to 50% does, so rare trophies count
as much as common ones). Rates are clamped to 0.01–99.99% before the logit,
as the scorer does.

1. **Log-time trend.** Fit `logit(rate) = a + b · ln(1 + hours since
   release)` by recency-weighted least squares (weights halve every 24 h)
   over the last 48 hours of reads (or all reads when fewer). Log-time is
   the natural clock for launch week: progress is fast at first and slows
   as the player base matures, and a line in log-time saturates gracefully
   instead of running away linearly.
2. **Tier shrinkage.** Slopes are shrunk toward their tier's mean slope
   (platinum / gold / silver / bronze), weighted by each fit's own standard
   error. Tiers progress at genuinely different speeds — bronze story
   trophies move first, platinum last — so the tier mean is a better prior
   than zero, and shrinkage keeps a noisy or flat trophy from extrapolating
   a fluke. A trophy whose history is perfectly flat still gets its tier's
   drift, not a confident zero.
3. **Anchored median.** The median starts at the last read's rate and moves
   by the shrunk slope over the horizon, with the total logit move capped
   at ±3. Anchoring (instead of reading the fitted intercept at the target)
   keeps the level honest to the freshest read; the slope only supplies the
   direction and size of the move.
4. **Honest spread.** The logit spread combines four terms in quadrature:
   - the house baseline's random walk, `0.5 · sqrt(days)`;
   - slope uncertainty × extrapolation distance (`se_b · Δx`);
   - the fit's residual volatility;
   - the median refresh jump between consecutive distinct reads (Exophase
     refreshes in steps, and a refresh can move every rate at once).
   The spread is floored at 0.30 and capped at 2.5 logit units, and widened
   further when the last rate is censored (printed as 0.00% or 100.00%).
   The pinball loss is a proper scoring rule: the forecast scores best by
   writing the quantiles it really believes, so the spread is deliberately
   wide over a 4.5-day horizon rather than tuned to look confident.
5. **Guardrails.** Quantiles are rounded to 4 decimals (half up, as the
   starter does), forced non-decreasing, and clamped to [0.01, 99.99].
   Any trophy that cannot be fitted (missing rates, degenerate history)
   falls back to its tier's typical rate with a wide spread; the program
   never crashes on a case — an invalid case forfeits the prize, so a
   valid forecast for every trophy always goes out.

## What the model does not do

- No network, no external data, no scraped trophy tables: the sandbox has
  no network and the program reads only the case file. GTA VI had no
  public trophy history before launch, so there is no prior game's curve to
  borrow — the forecast is built from this game's own reads.
- No player-count decomposition: forecasting earners and tracked players
  separately would double the extrapolation error, and the rates are
  printed to 2 decimals, so the derived earner counts of rare trophies are
  too noisy to fit. The logit trend already captures the net effect of
  dilution and progression.
- No tuning on the practice cases: their histories are flat and their
  answers are a single refresh, which says nothing about a week-long trend
  ("one refresh over 90 minutes is little guide to the final's week"). The
  practice cases are used only to check that the program reads a case and
  writes a valid forecast.

## Reproducing

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Deterministic: no randomness, no wall-clock reads, no dict-order
dependence — two runs on the same case write identical bytes.
