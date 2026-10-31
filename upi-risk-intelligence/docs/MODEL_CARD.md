# Model Card - UPI Behavioural Risk Model v2.0.0

**Intended use:** portfolio / demonstration system for behavioural fraud scoring with cost-based decisioning, conformal uncertainty and an analyst console. **Not** validated for production money.

**Model:** LightGBM (300 trees, defaults retained after a null tuning result) + Platt calibration + class-conditional conformal sets. Baselines: hand-written rules, logistic regression, Isolation Forest.

**Training data:** simulated UPI world, 91 days, ~741k payments (628k after a 14-day warm-up), ~1.6% fraud, 12k customers, 700 merchants in 13 categories, 12 cities, weekend / salary-week / festival effects,
account lifecycles (mule accounts rotate, ring accounts open just before use, genuine new customers keep joining so account age is stationary). Eight fraud typologies plus hard genuine look-alikes
(big one-off purchases, new phones, trips, bursts, night-shift users, shops receiving P2P, monthly rent, festival gifting). Methodology is additionally validated on the public ULB credit-card fraud dataset.

**Features (25):** amount z-score/ratio vs. own history, time since last payment, sender velocity (10m/1h/24h), receiver inflow (1h/24h), payee fan-in/out in 24h, geo-speed since last payment, new city / device / receiver,
just-under-limit amounts, round amounts, account ages, merchant category, hour / weekday, channel and payee type. No demographic attributes. Identical in batch (training) and streaming (serving) form - parity-tested.

**Evaluation:** chronological split (60/15/25), 4-fold time-series CV, cluster bootstrap over accounts, 5 independently generated worlds, cost-assumption sensitivity, label-delay / drift backtest,
conformal coverage, segment audit with significance testing. Results are in `reports/*.json`; narrative in the notebooks and README.

## Known limitations
* Simulated data and author-written fraud patterns; real adversaries adapt and real base rates are 10-100x lower. Six of eight typologies are recalled at ~100%, which reflects strong injected signatures.
* **First-leg blindness:** a per-transaction scorer cannot see a burst or escalation that has not started.
* **Concept drift:** a novel tactic cuts PR-AUC to ~0.60-0.68 for the first week regardless of retraining; label delay adds about a window; recovering the tactic needs alert capacity (23% / 58% / 78% recall at a 3% budget for static / delayed / oracle).
* **Monitoring gap:** score PSI never alarmed; feature PSI crossed 0.25 two weeks after onset; `new_receiver` drifts benignly as history accumulates. Needs outcome-based monitoring and bounded look-back windows.
* **Conformal under shift:** fraud coverage was 87.6% against a 90% target on the later test period.
* Cold-start accounts are scored with neutral history features (the API returns a warning).
* Cost figures are illustrative; the savings percentage is robust to them here (98.1-99.6%) only because simulated fraud amounts dwarf intervention costs.
* Segment attributes (age, city, app) are randomly assigned, so the fairness audit demonstrates method, not absence of bias.

## Operational & ethical notes
Wrongly blocking genuine users causes real harm: borderline cases go to a human REVIEW tier, conformal "uncertain" payments are surfaced for review, every decision is logged with reasons and model version
for appeals, analyst resolutions are stored as labels, and 11 CI gates fail the build on accuracy, calibration, drift, ring recovery, conformal coverage, segment disparity (significant and >2x) and latency
regressions. Retraining should follow a champion/challenger process with outcome-based monitoring.
