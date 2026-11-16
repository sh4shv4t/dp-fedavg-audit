# Presentation outline

Maps the five things the presentation has to cover to what is in this repo. Figures live in [`../figures`](../figures).

## 1. Problem identification and motivation

- FL keeps data on device, but the trained model can still leak whether a record was used (membership inference).
- DP-FedAvg promises protection through a single number, ε. Nobody hands you a way to check it.
- **Objectives**
  1. Train FedAvg and DP-FedAvg across noise levels on a realistic-ish setup (500 clients).
  2. Attack every model with a membership inference attack.
  3. Compare the claimed ε (upper bound) with the ε the attack can prove (lower bound).
  4. Measure the privacy vs accuracy trade-off.

## 2. Background and fundamental concepts

- FedAvg: local training, server averages updates.
- Differential privacy: (ε, δ) definition, smaller ε means stronger privacy.
- DP-FedAvg: per-client clipping (C) plus Gaussian noise (z · C) on the sum, Poisson client sampling (q).
- RDP accounting for the subsampled Gaussian mechanism turns (z, q, T, δ) into ε.
- Membership inference: loss-threshold attack (Yeom et al.), AUC, TPR at low FPR.
- Privacy auditing: DP implies `TPR ≤ e^ε · FPR + δ`, so a strong enough attack gives a lower bound on ε (Jagielski et al., Nasr et al.).

## 3. Proposed methodology and system design

- Pipeline diagram from the README "Method" section.
- Fixed data split: 10k members, 10k non-members, 10k test.
- 6 configs: FedAvg, Clip-only, z = 0.5, 1, 2, 4. 3 seeds each.
- Why these choices:
  - GroupNorm instead of BatchNorm (BatchNorm mixes samples, breaks per-client DP reasoning).
  - Fixed denominator `q · N` in aggregation so the sensitivity analysis holds.
  - Clip-only as a control to separate the effect of clipping from the effect of noise.
  - Clopper-Pearson plus Bonferroni so the empirical ε is a statistically valid lower bound, not a lucky threshold.

## 4. Implementation and experimental evaluation

- Stack: PyTorch, torchvision, Opacus (RDP accountant only), scikit-learn, SciPy, Colab T4.
- ~2.5 minutes per run, 18 runs total, resumable via saved `.npz` files.
- Metrics: test/train accuracy, AUC, TPR@1% and 10% FPR, max advantage, theoretical ε, empirical ε.
- Show: results table, `fig1_utility_vs_leakage.png`, `fig2_roc_and_loss_hist.png`, `fig2b_advantage_curve.png`, `fig3_audit_gap.png`.

## 5. Analysis, challenges and future work

- **Findings**
  - Noise lowers leakage monotonically, in the same order as ε.
  - Accuracy cost is steep with ~50 clients per round (54% to 16%).
  - Claimed ε 1.08 to 35.23 vs empirical ε ≈ 0: huge audit gap.
  - Even no-DP FedAvg leaks little because the model barely overfits (61% train vs 54% test).
- **Challenges**
  - Getting a valid lower bound needs confidence intervals, otherwise noise at low FPR looks like leakage.
  - Small client count makes DP noise dominate.
  - Colab disconnects, solved with Drive checkpoints and resumable runs.
- **Limitations**: one dataset, one model, IID, one weak attack, trusted server.
- **Future work**: LiRA, client-level canaries, white-box per-round attacker, more clients per round, non-IID data.
