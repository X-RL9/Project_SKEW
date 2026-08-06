# Skew — web interface

A Reflex (pure Python) website in front of the Skew fact-checking engine.
Paste a claim, get a verdict from one of six categories: Supported,
Contradicted, Mixed, Unproven, Insufficient Data, Not a Factual Claim.

## Status — read this before trusting anything it outputs

- The engine (classifier, source registry, statistical tests, verdict
  logic) is built and internally tested against synthetic data. See
  `pipeline/`, `classification/`, `registry/` for the code and their
  own docstrings for what's been verified vs. not.
- This web layer compiles and runs (backend boots, frontend builds
  clean) but **has never been tested with real users or real live data**
  — the dev sandbox this was built in has no network access to ONS or
  Home Office. The first real test of a live data fetch will happen
  once this is deployed somewhere with open network access. Expect it
  to fail the first few times (wrong dataset IDs, unexpected response
  shape) — that's the point of deploying it, to find out.
- Only immigration/employment and immigration/crime claims have real
  statistical logic behind them right now. Everything else (opinions,
  quotes, unrecognized claims) correctly reports itself as such rather
  than guessing.

## Running locally

```bash
pip install -r requirements.txt
reflex run
```

Then open the URL it prints (usually `http://localhost:3000`).

## Deploying (Railway)

1. Push this repo to GitHub.
2. In Railway: New Project → Deploy from GitHub repo → select this repo.
3. Railway auto-detects Python. Set the start command to:
   ```
   reflex run --env prod
   ```
   (Reflex needs both a frontend build step and a running backend —
   check Railway's current Reflex deployment guide if this needs
   adjusting, since exact platform config commands can change.)
4. Every push to the connected branch redeploys automatically — same
   workflow you're used to from Streamlit Community Cloud.

## Why pinned versions in requirements.txt

Every dependency here is pinned to an exact version (`==`, not `>=`).
This is the direct fix for the yfinance problem — an unpinned dependency
can change its behavior under you with no warning, including for anyone
who clones the repo later. If you need to upgrade a dependency
deliberately, do it as its own step, re-test, then commit the new pin —
never leave a range open by default.

## Project layout

```
rxconfig.py              # Reflex config
skew_web/skew_web.py      # the actual page + state (UI layer)
classification/           # claim classifier (unchanged from the engine build)
registry/                 # ONS / Home Office / Migration Observatory connectors
pipeline/                 # statistical tests + verdict engine
skew_pipeline.py           # SkewPipeline — wires the above into one call
requirements.txt           # pinned dependencies
```

## Next steps

1. Deploy to Railway, try a real claim, see what actually breaks against
   live ONS data — this is the real test that's been blocked so far.
2. Fix whatever that reveals (dataset IDs, response parsing).
3. Everything else (design pass, more claim patterns, more countries)
   comes after step 1 and 2 are solid.
