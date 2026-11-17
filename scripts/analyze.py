"""Turn results/raw into tables (results/*.csv, results/summary.json) and figures (figures/*.png)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dpfl_audit.accounting import clt_epsilon, rdp_epsilon  # noqa: E402
from dpfl_audit.attacks import (  # noqa: E402
    calibrated_loss_score,
    eps_lower_bound,
    lira_offline_score,
    loss_score,
    mia_metrics,
    roc,
)
from dpfl_audit.canary import CanaryResult  # noqa: E402
from dpfl_audit.config import PAPER_CONFIGS  # noqa: E402
from dpfl_audit.data import CLASSES  # noqa: E402
from dpfl_audit.io import exists, load_run, run_path  # noqa: E402

FIG, RES = Path("figures"), Path("results")
NAMES = list(PAPER_CONFIGS)
DP_NAMES = [n for n in NAMES if PAPER_CONFIGS[n].is_dp]
SEEDS = [0, 1, 2]
DELTA = 1e-5
COLORS = dict(zip(NAMES, ["#334155", "#64748b", "#38bdf8", "#0ea5e9", "#f97316", "#c2410c"]))

plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=160)
    plt.close(fig)
    print("wrote", FIG / name)


def theory(name):
    # the buggy run claims exactly what the correct z=2 run claims
    c = PAPER_CONFIGS[name.replace(" buggy", "")]
    if not c.is_dp:
        return float("inf"), float("inf")
    return rdp_epsilon(c.z, c.q, c.rounds, DELTA), clt_epsilon(c.z, c.q, c.rounds, DELTA)


# ------------------------------------------------------------------ loading
def load_kind(kind, names, seeds):
    out = {}
    for n in names:
        for s in seeds:
            p = run_path(kind, n, s)
            if exists(p):
                out[(n, s)] = load_run(p)
    return out


def attack_scores(main, refs, n, s):
    """Member / non-member scores for the three attacks on one target model."""
    _, a = main[(n, s)]
    r_ids = sorted(r for (m, r) in refs if m == n)
    out = {"Loss threshold": (loss_score(a["member_loss"]), loss_score(a["nonmember_loss"]))}
    if r_ids:
        rl_m = np.stack([refs[(n, r)][1]["member_loss"] for r in r_ids])
        rl_n = np.stack([refs[(n, r)][1]["nonmember_loss"] for r in r_ids])
        rp_m = np.stack([refs[(n, r)][1]["member_phi"] for r in r_ids])
        rp_n = np.stack([refs[(n, r)][1]["nonmember_phi"] for r in r_ids])
        out["Calibrated loss"] = (
            calibrated_loss_score(a["member_loss"], rl_m),
            calibrated_loss_score(a["nonmember_loss"], rl_n),
        )
        out["LiRA (offline)"] = (lira_offline_score(a["member_phi"], rp_m), lira_offline_score(a["nonmember_phi"], rp_n))
    return out


# ------------------------------------------------------------------ tables
def attack_table(main, refs):
    rows = []
    for n in NAMES:
        per = {}
        for s in SEEDS:
            if (n, s) not in main:
                continue
            meta, _ = main[(n, s)]
            for atk, (sm, sn) in attack_scores(main, refs, n, s).items():
                m = mia_metrics(sm, sn)
                m["eps_lb"] = eps_lower_bound(sm, sn, DELTA)
                m["test_acc"], m["train_acc"] = meta["accuracy"]["test"], meta["accuracy"]["member"]
                per.setdefault(atk, []).append(m)
        rdp, clt = theory(n)
        for atk, ms in per.items():
            df = pd.DataFrame(ms)
            row = dict(config=n, attack=atk, eps_rdp=rdp, eps_clt=clt, seeds=len(ms))
            for c in df.columns:
                row[c], row[c + "_std"] = df[c].mean(), df[c].std(ddof=0)
            rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(RES / "attacks.csv", index=False)
    return t


def canary_table(can):
    rows = []
    for n in NAMES + ["z=2 buggy"]:
        ests = []
        for s in SEEDS:
            if (n, s) not in can:
                continue
            meta, a = can[(n, s)]
            cr = CanaryResult(a["canary_final_obs"], a["canary_final_null"], a["canary_iter_obs"], a["canary_iter_null"])
            e = cr.estimate(DELTA)
            e["test_acc"] = meta["accuracy"]["test"]
            ests.append(e)
        if not ests:
            continue
        df = pd.DataFrame(ests)
        rdp, clt = theory(n)
        row = dict(config=n, eps_rdp=rdp, eps_clt=clt, seeds=len(ests))
        for c in df.columns:
            row[c], row[c + "_std"] = df[c].mean(), df[c].std(ddof=0)
        rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(RES / "canary_audit.csv", index=False)
    return t


# ------------------------------------------------------------------ figures
def fig_attacks(main, refs, at):
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
    styles = {"Loss threshold": ":", "Calibrated loss": "--", "LiRA (offline)": "-"}
    for n in ["FedAvg", "z=0.5", "z=4"]:
        if (n, 0) not in main:
            continue
        for atk, (sm, sn) in attack_scores(main, refs, n, 0).items():
            f, t = roc(sm, sn)
            ax[0].plot(f, t, styles[atk], color=COLORS[n], lw=1.6, label=f"{n} · {atk}")
    ax[0].plot([1e-4, 1], [1e-4, 1], color="gray", lw=0.8)
    ax[0].set(xscale="log", yscale="log", xlim=(1e-4, 1), ylim=(1e-4, 1), xlabel="False positive rate",
              ylabel="True positive rate", title="Attack ROC, log-log (seed 0)")
    ax[0].legend(fontsize=7, ncol=1, frameon=False)

    w = 0.27
    x = np.arange(len(NAMES))
    for i, atk in enumerate(styles):
        sub = at[at.attack == atk].set_index("config").reindex(NAMES)
        ax[1].bar(x + (i - 1) * w, sub["tpr_1"] * 100, w, yerr=sub["tpr_1_std"] * 100, label=atk,
                  color=["#94a3b8", "#38bdf8", "#f97316"][i], capsize=2)
    ax[1].axhline(1, color="gray", ls="--", lw=1)
    ax[1].text(len(NAMES) - 0.5, 1.02, "random guessing (1%)", ha="right", va="bottom", fontsize=8, color="gray")
    ax[1].set_xticks(x, NAMES)
    ax[1].set(ylabel="TPR at 1% FPR (%)", title="True positives at a strict 1% false-alarm rate")
    ax[1].legend(frameon=False, fontsize=8)
    save(fig, "fig4_attack_comparison.png")


def fig_epsilon_ladder(at, ct):
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    x = np.arange(len(DP_NAMES))
    series = [
        ("Claimed: RDP accountant", [theory(n)[0] for n in DP_NAMES], "#0f172a"),
        ("Approx: GDP central limit", [theory(n)[1] for n in DP_NAMES], "#64748b"),
    ]
    if len(ct):
        c = ct.set_index("config").reindex(DP_NAMES)
        series += [
            ("Canary, all rounds visible (estimate)", c["iterates_eps_est"].values, "#c2410c"),
            ("Canary, final model only (estimate)", c["final_eps_est"].values, "#f97316"),
            ("Canary, final model (95% lower bound)", c["final_eps_lb"].values, "#fdba74"),
        ]
    lira = at[at.attack == "LiRA (offline)"].set_index("config").reindex(DP_NAMES)
    if len(lira.dropna()):
        series.append(("LiRA on records (95% lower bound)", lira["eps_lb"].values, "#38bdf8"))
    w = 0.8 / len(series)
    for i, (lab, vals, col) in enumerate(series):
        b = ax.bar(x + (i - (len(series) - 1) / 2) * w, vals, w, label=lab, color=col)
        for r, v in zip(b, vals):
            ax.annotate(f"{v:.2f}", (r.get_x() + r.get_width() / 2, max(v, 0)), ha="center", va="bottom", fontsize=6.5)
    ax.set_yscale("symlog", linthresh=0.1)
    ax.set_ylim(0, None)
    ax.set_xticks(x, DP_NAMES)
    ax.set(ylabel="epsilon (symlog)", title="The epsilon ladder: what is promised vs what each auditor can see")
    ax.legend(fontsize=7.5, frameon=False, ncol=2)
    save(fig, "fig5_epsilon_ladder.png")


def fig_canary_hist(can):
    show = [n for n in ["FedAvg", "z=0.5", "z=2"] if (n, 0) in can]
    fig, ax = plt.subplots(1, len(show), figsize=(4.3 * len(show), 3.6), squeeze=False)
    for axi, n in zip(ax[0], show):
        a = can[(n, 0)][1]
        null, obs = a["canary_iter_null"], a["canary_iter_obs"]
        bins = np.linspace(min(null.min(), obs.min()), max(null.max(), obs.max()), 45)
        axi.hist(null, bins, density=True, alpha=0.6, color="#94a3b8", label="never inserted")
        axi.hist(obs, bins, density=True, alpha=0.7, color="#f97316", label="canary clients")
        axi.set(title=n, xlabel="max cosine with a round update")
        axi.legend(frameon=False, fontsize=8)
    ax[0][0].set_ylabel("density")
    save(fig, "fig6_canary_separation.png")


def fig_dynamics(main):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for n in NAMES:
        if (n, 0) not in main:
            continue
        meta, a = main[(n, 0)]
        r = np.arange(1, len(a["diag_signal"]) + 1)
        if PAPER_CONFIGS[n].is_dp:
            snr = a["diag_signal"] / a["diag_noise"]
            ax[0].plot(r, pd.Series(snr).rolling(5, min_periods=1).mean(), color=COLORS[n], label=n)
        h = pd.DataFrame(meta["history"])
        h = h[h["round"] > 0]
        ax[1].plot(h["round"], h["auc"], "o-", ms=3, color=COLORS[n], label=n)
        ax[2].plot(h["round"], h["train_acc"] - h["test_acc"], "o-", ms=3, color=COLORS[n], label=n)
    ax[0].axhline(1, color="gray", ls="--", lw=1)
    ax[0].set(yscale="log", xlabel="round", ylabel="||sum of clipped updates|| / ||noise||",
              title="Signal-to-noise of each round's aggregate")
    ax[1].axhline(0.5, color="gray", ls="--", lw=1)
    ax[1].set(xlabel="round", ylabel="loss-attack AUC", title="Leakage builds up during training")
    ax[2].set(xlabel="round", ylabel="train acc − test acc", title="Generalisation gap")
    for a_ in ax:
        a_.legend(fontsize=7, frameon=False)
    save(fig, "fig7_training_dynamics.png")


def fig_update_norms(main):
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for n in ["FedAvg", "Clip-only", "z=1", "z=4"]:
        if (n, 0) not in main:
            continue
        a = main[(n, 0)][1]
        ax.plot(a["diag_norm_med"], color=COLORS[n], label=f"{n} (median)")
        ax.fill_between(np.arange(len(a["diag_norm_med"])), a["diag_norm_med"], a["diag_norm_max"], color=COLORS[n], alpha=0.12)
    ax.axhline(1.0, color="#c2410c", ls="--", lw=1)
    ax.text(2, 1.03, "clip norm C = 1", color="#c2410c", fontsize=8)
    ax.set(xlabel="round", ylabel="client update L2 norm (median to max)", title="How big are client updates before clipping?")
    ax.legend(fontsize=8, frameon=False)
    save(fig, "fig8_update_norms.png")


def fig_vulnerable(main, refs, split):
    """Which training images leak the most under FedAvg, according to LiRA."""
    if ("FedAvg", 0) not in main or not any(m == "FedAvg" for m, _ in refs):
        return None
    sm, sn = attack_scores(main, refs, "FedAvg", 0)["LiRA (offline)"]
    order = np.argsort(-sm)
    fig, ax = plt.subplots(2, 8, figsize=(13, 4.4), gridspec_kw=dict(hspace=0.45))
    idx = split.member_idx
    for row, (title, picks) in enumerate([("most exposed", order[:8]), ("least exposed", order[-8:])]):
        for j, k in enumerate(picks):
            img = split.denormalise(split.X[idx[k]]).permute(1, 2, 0).numpy()
            ax[row, j].imshow(img, interpolation="nearest")
            ax[row, j].set_title(f"{CLASSES[int(split.Y[idx[k]])]}\nscore {sm[k]:.1f}", fontsize=8)
            ax[row, j].axis("off")
        ax[row, 0].text(-12, 16, title, rotation=90, va="center", ha="center", fontsize=10)
    fig.suptitle("FedAvg members ranked by LiRA score: atypical images leak first", fontsize=11)
    save(fig, "fig9_vulnerable_examples.png")

    # per-class leakage
    ym = split.Y[split.member_idx].numpy()
    yn = split.Y[split.nonmember_idx].numpy()
    per = []
    for c in range(10):
        per.append(dict(cls=CLASSES[c], **{f"auc_{k}": mia_metrics(v[0][ym == c], v[1][yn == c])["auc"]
                                           for k, v in attack_scores(main, refs, "FedAvg", 0).items()}))
    pc = pd.DataFrame(per)
    pc.to_csv(RES / "per_class_leakage.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    srt = pc.sort_values("auc_LiRA (offline)")
    ax.barh(srt.cls, srt["auc_LiRA (offline)"] - 0.5, left=0.5, color="#f97316", label="LiRA")
    ax.scatter(srt["auc_Loss threshold"], srt.cls, color="#0f172a", zorder=3, s=18, label="loss threshold")
    ax.axvline(0.5, color="gray", ls="--", lw=1)
    ax.set(xlabel="attack AUC within the class", title="Per-class leakage under FedAvg (seed 0)")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "fig10_per_class_leakage.png")

    # difficulty vs exposure: what LiRA is actually picking up
    r_ids = sorted(r for (m, r) in refs if m == "FedAvg")
    ref_loss_m = np.stack([refs[("FedAvg", r)][1]["member_loss"] for r in r_ids]).mean(0)
    ref_loss_n = np.stack([refs[("FedAvg", r)][1]["nonmember_loss"] for r in r_ids]).mean(0)
    _, a = main[("FedAvg", 0)]
    bins = np.quantile(np.r_[ref_loss_m, ref_loss_n], np.linspace(0, 1, 11))
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mm, mn = (ref_loss_m >= lo) & (ref_loss_m <= hi), (ref_loss_n >= lo) & (ref_loss_n <= hi)
        rows.append(dict(
            difficulty=(lo + hi) / 2,
            auc_loss=mia_metrics(-a["member_loss"][mm], -a["nonmember_loss"][mn])["auc"],
            auc_lira=mia_metrics(sm[mm], sn[mn])["auc"],
        ))
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(df.difficulty, df.auc_lira, "o-", color="#f97316", label="LiRA")
    ax.plot(df.difficulty, df.auc_loss, "o-", color="#0f172a", label="loss threshold")
    ax.axhline(0.5, color="gray", ls="--", lw=1)
    ax.set(xscale="log", xlabel="difficulty: mean loss on reference models that never saw the image",
           ylabel="attack AUC within the bin", title="Hard images leak more")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "fig11_difficulty_vs_leakage.png")
    return pc


def scaling_table():
    rows = []
    for n_cl in [500, 1000, 2500, 5000]:
        for name in ["FedAvg", "z=1", "z=2"]:
            p = run_path("scaling", f"{name}_N{n_cl}", 0)
            if not exists(p):
                continue
            meta, a = load_run(p)
            c = PAPER_CONFIGS[name]
            rows.append(dict(
                config=name, n_clients=n_cl, per_round=c.q * n_cl, records_each=10_000 // n_cl,
                eps_rdp=rdp_epsilon(c.z, c.q, c.rounds, DELTA) if c.is_dp else float("inf"),
                test_acc=meta["accuracy"]["test"],
                auc=mia_metrics(-a["member_loss"], -a["nonmember_loss"])["auc"],
                noise_std_on_mean=(c.z * 1.0 / (c.q * n_cl)) if c.is_dp else 0.0,
            ))
    t = pd.DataFrame(rows)
    if len(t):
        t.to_csv(RES / "scaling.csv", index=False)
        fig, ax = plt.subplots(figsize=(6.8, 4))
        for name, col in [("FedAvg", "#334155"), ("z=1", "#0ea5e9"), ("z=2", "#f97316")]:
            s = t[t.config == name]
            lab = name if name == "FedAvg" else f"{name} (ε = {s.eps_rdp.iloc[0]:.2f}, constant)"
            ax.plot(s.per_round, s.test_acc * 100, "o-", color=col, label=lab)
        ax.set(xscale="log", xlabel="clients per round (total data fixed at 10k records)", ylabel="test accuracy (%)",
               title="Same privacy budget, more clients: the DP tax shrinks")
        ax.set_xticks([50, 100, 250, 500], ["50", "100", "250", "500"])
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.legend(frameon=False, fontsize=8)
        save(fig, "fig12_scaling.png")
    return t


def main():
    from dpfl_audit.data import make_split

    FIG.mkdir(exist_ok=True)
    main_runs = load_kind("main", NAMES, SEEDS)
    refs = load_kind("refs", NAMES, range(16))
    can = load_kind("canary", NAMES + ["z=2 buggy"], SEEDS)
    print(f"loaded {len(main_runs)} main, {len(refs)} reference, {len(can)} canary runs")

    at = attack_table(main_runs, refs)
    ct = canary_table(can)
    fig_attacks(main_runs, refs, at)
    fig_epsilon_ladder(at, ct)
    if can:
        fig_canary_hist(can)
    fig_dynamics(main_runs)
    fig_update_norms(main_runs)
    fig_vulnerable(main_runs, refs, make_split())
    st = scaling_table()

    summary = dict(
        attacks=at.round(4).to_dict("records"),
        canary=ct.round(4).to_dict("records") if len(ct) else [],
        scaling=st.round(4).to_dict("records") if len(st) else [],
    )
    (RES / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        cols = ["config", "attack", "test_acc", "auc", "tpr_01", "tpr_1", "eps_lb", "eps_rdp"]
        print(at[cols].round(4).to_string(index=False))
        if len(ct):
            print(ct[["config", "eps_rdp", "eps_clt", "iterates_mu", "iterates_eps_est", "final_mu", "final_eps_est",
                      "final_eps_lb", "iterates_eps_lb", "test_acc"]].round(3).to_string(index=False))
        if len(st):
            print(st.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
