# Bounty-GTA-6-Trophy-Forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/egilolriot/Bounty-GTA-6-Trophy-Forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/egilolriot/Bounty-GTA-6-Trophy-Forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecast how many GTA VI players earn each trophy in week one — a per-trophy log-time logit trend with tier shrinkage and an uncertainty-aware spread, built from the game's own Exophase reads. See [METHOD.md](METHOD.md) for the full reasoning.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` (Python 3.13, standard library only) reads every `<id>.in` case file in the input directory and writes `<id>.out` to the output directory: one row per trophy with five quantiles (5 / 25 / 50 / 75 / 95%) of its Exophase earn rate at the case's target. No network, no randomness — two runs on the same case write identical bytes.

---

<sub>Entered on [make.fun](https://make.fun/?ref=zegsd6t5&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
