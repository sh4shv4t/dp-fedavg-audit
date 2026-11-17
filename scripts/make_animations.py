"""Explainer GIFs for the README (assets/*.gif).

    python scripts/make_animations.py            # all
    python scripts/make_animations.py mechanism  # just one

mechanism   how one round of DP-FedAvg works (sampling, clipping, summing, noise)
attack      how a membership inference attack works, on real loss snapshots
canary      how the canary audit turns cosines into an epsilon, on real canary runs
usecases    where this matters in the real world
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dpfl_audit.accounting import gdp_epsilon  # noqa: E402

BG, PANEL, EDGE = "#0f172a", "#1e293b", "#334155"
TEXT, MUTED = "#e2e8f0", "#94a3b8"
SKY, ORANGE, RED, GREEN = "#38bdf8", "#f97316", "#ef4444", "#22c55e"
OUT = Path("assets")
plt.rcParams.update({"font.family": "DejaVu Sans"})


# ------------------------------------------------------------------ helpers
def ease(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def ph(f, start, length):
    return float(ease((f - start) / length))


def mix(a, b, t):
    a, b = np.array(to_rgb(a)), np.array(to_rgb(b))
    return tuple(a + (b - a) * t)


def blank(*axes):
    for ax in axes:
        ax.clear()
        ax.set_facecolor(BG)
        ax.axis("off")


def arrow(ax, start, end, color, lw=2.2, ms=13, alpha=1.0, z=3):
    if np.linalg.norm(np.subtract(end, start)) < 1e-3:
        return
    ax.annotate("", xy=end, xytext=start, zorder=z,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=ms, alpha=alpha, shrinkA=0, shrinkB=0))


def box(ax, xy, w, h, ec=EDGE, fc=PANEL, lw=1.6, r=0.12):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle=f"round,pad=0.02,rounding_size={r}", fc=fc, ec=ec, lw=lw))


def caption(ax, step, text, f=None):
    ax.text(0.025, 0.5, step, color=ORANGE, fontsize=12.5, weight="bold", va="center", transform=ax.transAxes)
    ax.text(0.215, 0.5, text, color=TEXT, fontsize=10.5, va="center", transform=ax.transAxes, wrap=True)


def save(fig, update, frames, name, fps=14):
    OUT.mkdir(exist_ok=True)
    FuncAnimation(fig, update, frames=frames).save(OUT / name, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print("wrote", OUT / name, f"{(OUT / name).stat().st_size / 1e6:.1f} MB")


# ------------------------------------------------------------------ 1. mechanism
def mechanism():
    rng = np.random.default_rng(3)
    ys = np.linspace(4.45, 0.55, 10)
    sel = [1, 4, 6, 8]
    raw = np.array([(2.6, 0.8), (0.5, 1.9), (-0.6, 0.7), (1.8, -2.2)])
    C = 1.2
    clipped = raw * np.minimum(1, C / np.linalg.norm(raw, axis=1))[:, None]
    offs = np.vstack([[0, 0], np.cumsum(clipped, 0)[:-1]])
    total = clipped.sum(0)
    cloud = rng.normal(0, C, (260, 2))
    noise = np.array([0.95, -0.75])
    phases = [
        (0, "1 · Sample", "Each round every client joins with probability q. Here 4 of 10 do."),
        (22, "2 · Train", "Selected clients train on their own data and send an update, never the data."),
        (48, "3 · Clip", "The server caps each update at length C. Nobody gets to shout louder."),
        (74, "4 · Sum", "Clipped updates are added up, tip to tail."),
        (98, "5 · Noise", "Gaussian noise with std z·C goes on the sum. One client's arrow is now lost in it."),
        (130, "6 · Share", "The noisy average becomes the new global model and goes back to everyone."),
    ]
    fig = plt.figure(figsize=(10, 4.6), dpi=88, facecolor=BG)
    aL = fig.add_axes([0.02, 0.15, 0.5, 0.8])
    aR = fig.add_axes([0.56, 0.15, 0.42, 0.8])
    aC = fig.add_axes([0, 0, 1, 0.13])

    def update(f):
        blank(aL, aR, aC)
        _, step, txt = max((p for p in phases if p[0] <= f), key=lambda p: p[0])
        caption(aC, step, txt)

        # ---- left: clients and server
        aL.set_xlim(0, 10)
        aL.set_ylim(0, 5)
        on = ph(f, 2, 12)
        for i, y in enumerate(ys):
            ec = mix(EDGE, ORANGE, on) if i in sel else EDGE
            box(aL, (0.55, y - 0.18), 0.55, 0.36, ec=ec, lw=2)
            for k in range(3):
                aL.add_patch(Circle((0.7 + k * 0.13, y), 0.035, color=MUTED if i not in sel else mix(MUTED, ORANGE, on)))
        aL.text(0.82, 4.85, "clients (private data stays here)", color=MUTED, fontsize=9, ha="left")
        box(aL, (6.75, 1.75), 2.9, 1.5, ec=SKY, lw=2)
        aL.text(8.2, 2.75, "server", color=TEXT, ha="center", fontsize=11, weight="bold")
        rnd = "w(t+1)" if f >= 146 else "w(t)"
        aL.text(8.2, 2.25, f"global model {rnd}", color=SKY, ha="center", fontsize=9.5)
        t_up = ph(f, 24, 18)
        if 22 <= f < 48:
            for i in sel:
                p = (1.15 + t_up * (6.7 - 1.15), ys[i] + t_up * (2.5 - ys[i]))
                aL.plot([1.15, p[0]], [ys[i], p[1]], color=ORANGE, lw=1, alpha=0.35)
                aL.add_patch(Circle(p, 0.09, color=ORANGE))
        if f >= 130:
            t_dn = ph(f, 132, 16)
            for y in ys:
                p = (6.7 + t_dn * (1.15 - 6.7), 2.5 + t_dn * (y - 2.5))
                if t_dn < 0.999:
                    aL.add_patch(Circle(p, 0.07, color=SKY))

        # ---- right: update space
        aR.set_xlim(-3.0, 4.6)
        aR.set_ylim(-2.9, 3.0)
        aR.set_aspect("equal")
        aR.text(-2.95, 2.75, "update space (a 2-D sketch of a 67k-D vector)", color=MUTED, fontsize=8.5)
        aR.plot([-2.8, 4.4], [0, 0], color=EDGE, lw=0.8)
        aR.plot([0, 0], [-2.7, 2.6], color=EDGE, lw=0.8)
        if f >= 22:
            g = ph(f, 24, 18)
            c = ph(f, 54, 16)
            s = ph(f, 76, 18)
            vecs = raw + (clipped - raw) * c
            if f >= 48:
                a = ph(f, 48, 8)
                aR.add_patch(Circle((0, 0), C * a, fill=False, ec=SKY, ls="--", lw=1.4, alpha=0.9))
                aR.text(C * 0.72, -C * 0.95, "C", color=SKY, fontsize=11, alpha=a)
            for v, o in zip(vecs, offs):
                st = o * s
                arrow(aR, st, st + v * g, ORANGE if f < 98 else mix(ORANGE, EDGE, ph(f, 100, 12)), lw=2)
            if f >= 90:
                arrow(aR, (0, 0), total * ph(f, 90, 8), TEXT, lw=2.8)
        if f >= 98:
            k = ph(f, 100, 14)
            pts = total + cloud * k
            aR.scatter(pts[:, 0], pts[:, 1], s=4, color=RED, alpha=0.28 * min(1, 3 * k), lw=0)
            if f >= 114:
                tn = ph(f, 114, 10)
                arrow(aR, total, total + noise * tn, RED, lw=2)
                if tn > 0.99:
                    arrow(aR, (0, 0), total + noise, SKY, lw=3.2, ms=16, z=5)
                    aR.text(*(total + noise + [0.1, -0.35]), "released update", color=SKY, fontsize=9)

    save(fig, update, range(166), "dp_fedavg_mechanism.gif", fps=12)


# ------------------------------------------------------------------ 2. attack (real data)
def attack():
    from dpfl_audit.io import exists, load_run, run_path

    runs = {}
    for name in ["FedAvg", "z=4"]:
        p = run_path("main", name, 0)
        if not exists(p):
            print("skip attack.gif: no snapshots yet")
            return
        runs[name] = load_run(p)[1]
    rounds = runs["FedAvg"]["snap_rounds"]
    bins = np.linspace(0, 4.5, 46)
    steps_per = 6
    n_train = (len(rounds) - 1) * steps_per + 1
    n_sweep = 50
    sweep = np.r_[np.linspace(4.0, 0.15, n_sweep // 2), np.linspace(0.15, 1.1, n_sweep // 2)]

    def hist(a, i):
        return np.histogram(np.clip(a, 0, 4.49), bins, density=True)[0]

    H = {n: [(hist(r["snap_member"][i], i), hist(r["snap_nonmember"][i], i)) for i in range(len(rounds))] for n, r in runs.items()}
    fig = plt.figure(figsize=(10, 4.6), dpi=88, facecolor=BG)
    axes = [fig.add_axes([0.05, 0.25, 0.42, 0.62]), fig.add_axes([0.55, 0.25, 0.42, 0.62])]
    aC = fig.add_axes([0, 0, 1, 0.13])
    centers = (bins[:-1] + bins[1:]) / 2

    def update(f):
        blank(aC)
        if f < n_train:
            seg, t = divmod(f, steps_per)
            t = t / steps_per
            j = min(seg + 1, len(rounds) - 1)
            rnd = rounds[seg] + (rounds[j] - rounds[seg]) * t
            thr = None
            caption(aC, f"Round {rnd:>3.0f}", "The model sees its training images every round, so their loss (orange) drops a bit faster than outsiders' (grey).")
        else:
            seg, t, j, rnd = len(rounds) - 1, 0, len(rounds) - 1, rounds[-1]
            thr = sweep[min(f - n_train, n_sweep - 1)]
            caption(aC, "The attack", "Pick a loss threshold. Below it, guess 'member'. Gap between the two rates = what leaked.")
        for ax, (name, r) in zip(axes, runs.items()):
            ax.clear()
            ax.set_facecolor(BG)
            hm = H[name][seg][0] + (H[name][j][0] - H[name][seg][0]) * t
            hn = H[name][seg][1] + (H[name][j][1] - H[name][seg][1]) * t
            ax.fill_between(centers, hn, step="mid", color=MUTED, alpha=0.45, label="non-members")
            ax.fill_between(centers, hm, step="mid", color=ORANGE, alpha=0.55, label="members (training data)")
            ax.set_ylim(0, 1.6)
            ax.set_xlim(0, 4.5)
            ax.set_title(name + ("  (no privacy)" if name == "FedAvg" else "  (strongest noise)"), color=TEXT, fontsize=11)
            ax.set_xlabel("per-example loss", color=MUTED)
            ax.tick_params(colors=MUTED, labelsize=8)
            for s_ in ax.spines.values():
                s_.set_color(EDGE)
            if thr is not None:
                m, n = r["snap_member"][-1], r["snap_nonmember"][-1]
                tpr, fpr = (m < thr).mean(), (n < thr).mean()
                ax.axvline(thr, color=SKY, lw=2)
                ax.axvspan(0, thr, color=SKY, alpha=0.06)
                ax.text(0.98, 0.92, f"caught members  {tpr:5.1%}\nfalse alarms    {fpr:5.1%}\nadvantage       {tpr - fpr:+5.1%}",
                        transform=ax.transAxes, ha="right", va="top", color=TEXT, fontsize=9, family="DejaVu Sans Mono",
                        bbox=dict(fc=PANEL, ec=EDGE))
            ax.legend(loc="upper left", fontsize=8, frameon=False, labelcolor=TEXT)

    save(fig, update, range(n_train + n_sweep + 8), "membership_attack.gif", fps=12)


# ------------------------------------------------------------------ 3. canary (real data)
def canary():
    from dpfl_audit.io import exists, load_run, run_path

    names = ["FedAvg", "Clip-only", "z=0.5", "z=1", "z=2", "z=4"]
    data = {}
    for n in names:
        p = run_path("canary", n, 0)
        if exists(p):
            data[n] = load_run(p)[1]
    if len(data) < 3:
        print("skip canary.gif: canary runs missing")
        return
    names = [n for n in names if n in data]
    stats = []
    for n in names:
        a = data[n]
        null, obs = a["canary_iter_null"], a["canary_iter_obs"]
        z = (obs - null.mean()) / null.std()
        zn = (null - null.mean()) / null.std()
        mu = float(z.mean())
        stats.append((n, zn, z, mu, gdp_epsilon(mu, 1e-5)))
    hold = 26
    fig = plt.figure(figsize=(10, 4.6), dpi=88, facecolor=BG)
    aL = fig.add_axes([0.02, 0.17, 0.36, 0.78])
    aR = fig.add_axes([0.45, 0.27, 0.52, 0.6])
    aC = fig.add_axes([0, 0, 1, 0.13])
    rng = np.random.default_rng(0)
    others = rng.normal(0, 1, (7, 2)) * 0.9
    bins = np.linspace(-4, 30, 69)

    def update(f):
        blank(aL, aC)
        k, t = divmod(f, hold)
        k = min(k, len(stats) - 1)
        n, zn, zo, mu, eps = stats[k]
        caption(aC, "Canary audit", "A fake client always sends the same random direction. We check how far it sticks out of what the server released.")
        aL.set_xlim(-3, 3)
        aL.set_ylim(-3, 3.2)
        aL.set_aspect("equal")
        aL.text(-2.9, 2.95, "inside one round", color=MUTED, fontsize=9)
        for v in others:
            arrow(aL, (0, 0), v, EDGE, lw=1.6)
        arrow(aL, (0, 0), (1.3, 1.3), ORANGE, lw=2.6)
        aL.text(1.35, 1.45, "canary", color=ORANGE, fontsize=10)
        noise_r = {"FedAvg": 0, "Clip-only": 0, "z=0.5": 0.8, "z=1": 1.4, "z=2": 2.1, "z=4": 2.8}.get(n, 1)
        if noise_r:
            pts = np.random.default_rng(f).normal(0, noise_r / 2, (220, 2))
            aL.scatter(pts[:, 0], pts[:, 1], s=4, color=RED, alpha=0.35, lw=0)
        aL.text(0, -2.85, f"{n}", color=TEXT, fontsize=13, ha="center", weight="bold")

        aR.clear()
        aR.set_facecolor(BG)
        aR.hist(zn, bins, density=True, color=MUTED, alpha=0.55, label="never inserted")
        aR.hist(np.clip(zo, -4, 29.9), bins, density=True, color=ORANGE, alpha=0.75, label="canary clients")
        aR.axvline(0, color=MUTED, lw=1, ls=":")
        aR.axvline(min(mu, 29.5), color=ORANGE, lw=1.5, ls="--")
        aR.set_xlim(-4, 30)
        aR.set_ylim(0, 0.75)
        aR.set_xlabel("how much the canary sticks out (std devs above chance)", color=MUTED)
        aR.tick_params(colors=MUTED, labelsize=8)
        for s_ in aR.spines.values():
            s_.set_color(EDGE)
        aR.legend(loc="upper right", fontsize=8.5, frameon=False, labelcolor=TEXT)
        eps_s = f"{eps:.1f}" if eps < 1000 else "∞"
        aR.text(0.98, 0.62, f"shift μ = {mu:.2f}\nmeasured ε ≈ {eps_s}", transform=aR.transAxes, ha="right", va="top",
                color=TEXT, fontsize=11, family="DejaVu Sans Mono", bbox=dict(fc=PANEL, ec=ORANGE))

    save(fig, update, range(hold * len(stats)), "canary_audit.gif", fps=10)


# ------------------------------------------------------------------ 4. use cases
def usecases():
    import pandas as pd

    bug = None
    p = Path("results/canary_audit.csv")
    if p.exists():
        t = pd.read_csv(p).set_index("config")
        if "z=2 buggy" in t.index and "z=2" in t.index:
            bug = (t.loc["z=2", "eps_rdp"], t.loc["z=2", "iterates_eps_lb"], t.loc["z=2 buggy", "iterates_eps_lb"])
    scene_len = 70
    fig = plt.figure(figsize=(10, 4.6), dpi=88, facecolor=BG)
    ax = fig.add_axes([0, 0.13, 1, 0.87])
    aC = fig.add_axes([0, 0, 1, 0.13])

    def typed(s, t):
        return s[: int(len(s) * t)]

    def scene_keyboard(f):
        ax.text(0.5, 9.3, "Phone keyboards  ·  cross-device FL", color=TEXT, fontsize=15, weight="bold", ha="center")
        for i in range(6):
            x = 0.9 + i * 1.45
            box(ax, (x, 5.6), 0.8, 1.5, ec=SKY if i % 2 == 0 else EDGE, lw=1.8)
            ax.add_patch(Circle((x + 0.4, 5.8), 0.06, color=MUTED))
        t = ph(f, 4, 18)
        for i in range(6):
            x = 1.3 + i * 1.45
            ax.add_patch(Circle((x + t * (5 - x), 7.2 + t * 0.9), 0.09, color=ORANGE, alpha=1 - 0.6 * t))
        ax.text(5, 8.4, "next-word model", color=SKY, ha="center", fontsize=10)
        ax.text(0.6, 4.5, 'Someone asks the trained model to finish:  "my card number is …"', color=TEXT, fontsize=11)
        t2 = ph(f, 26, 20)
        box(ax, (0.6, 1.4), 4.1, 2.4, ec=RED)
        ax.text(0.8, 3.3, "without DP", color=RED, fontsize=10, weight="bold")
        ax.text(0.8, 2.3, typed("4417 2390 8812 …", t2), color=TEXT, fontsize=14, family="DejaVu Sans Mono")
        ax.text(0.8, 1.65, "one user's secret, memorised" if t2 > 0.99 else "", color=MUTED, fontsize=9)
        box(ax, (5.3, 1.4), 4.1, 2.4, ec=GREEN)
        ax.text(5.5, 3.3, "with DP-FedAvg", color=GREEN, fontsize=10, weight="bold")
        ax.text(5.5, 2.3, typed("on the back of …", t2), color=TEXT, fontsize=14, family="DejaVu Sans Mono")
        ax.text(5.5, 1.65, "only patterns many users share" if t2 > 0.99 else "", color=MUTED, fontsize=9)
        caption(aC, "Real example", "Gboard trains its language models with DP federated learning, so one person's rare phrase cannot be learned.")

    def scene_hospital(f):
        ax.text(5, 9.3, "Hospitals  ·  cross-silo FL", color=TEXT, fontsize=15, weight="bold", ha="center")
        for i, (x, y) in enumerate([(0.5, 6.4), (0.5, 3.6), (2.6, 6.4), (2.6, 3.6)]):
            box(ax, (x, y), 1.7, 1.9, ec=SKY, lw=1.8)
            ax.text(x + 0.85, y + 1.2, "+", color=RED, fontsize=20, ha="center", weight="bold")
            ax.text(x + 0.85, y + 0.35, f"Hospital {'ABCD'[i]}", color=TEXT, fontsize=8.5, ha="center")
        ax.text(0.5, 2.6, 'Question: "was patient #2731 in\nthe oncology training set?"', color=TEXT, fontsize=10, va="top")
        ax.text(0.5, 1.2, "A yes reveals the diagnosis.", color=ORANGE, fontsize=10)
        ax.text(5.3, 8.2, "Best an attacker can ever be (50% prior):", color=MUTED, fontsize=10)
        rows = [("no DP", float("inf")), ("ε = 7.90", 7.90), ("ε = 2.58", 2.58), ("ε = 1.08", 1.08)]
        for k, (lab, e) in enumerate(rows):
            y = 7.0 - k * 1.45
            cap = 1.0 if not np.isfinite(e) else np.exp(e) / (1 + np.exp(e))
            g = ph(f, 8 + 8 * k, 14)
            ax.text(5.3, y + 0.25, lab, color=TEXT, fontsize=10, family="DejaVu Sans Mono")
            box(ax, (6.4, y), 2.7, 0.55, ec=EDGE, fc=PANEL, r=0.05)
            col = RED if cap > 0.95 else (ORANGE if cap > 0.85 else GREEN)
            ax.add_patch(FancyBboxPatch((6.4, y), 2.7 * cap * g, 0.55, boxstyle="round,pad=0,rounding_size=0.05", fc=col, ec="none"))
            ax.text(9.2, y + 0.27, f"{cap * g:.1%}", color=TEXT, fontsize=9.5, va="center")
        caption(aC, "The math", "(ε, δ)-DP caps an attacker's odds: posterior ≤ e^ε × prior. That is what the number buys you.")

    def scene_audit(f):
        ax.text(5, 9.3, "Any FL operator  ·  privacy audit as a release gate", color=TEXT, fontsize=15, weight="bold", ha="center")
        steps = ["train with\ncanary clients", "measure how\nmuch they leak", "compare with\nclaimed ε", "ship or block"]
        for i, s in enumerate(steps):
            x = 0.4 + i * 2.45
            on = ph(f, 4 + 9 * i, 8)
            box(ax, (x, 5.8), 2.0, 1.7, ec=mix(EDGE, SKY, on), lw=2)
            ax.text(x + 1.0, 6.65, s, color=mix(MUTED, TEXT, on), fontsize=10, ha="center", va="center")
            if i < 3:
                arrow(ax, (x + 2.05, 6.65), (x + 2.4, 6.65), mix(EDGE, SKY, on), lw=1.8)
        if bug is not None:
            claim, ok_lb, bug_lb = bug
            lines = [(f"correct z=2:  claimed ε {claim:.2f}, audit proves ≥ {ok_lb:.2f}", GREEN, "ship ✓"),
                     (f"buggy   z=2:  claimed ε {claim:.2f}, audit proves ≥ {bug_lb:.2f}", RED, "block ✗")]
        else:
            lines = [("correct implementation: measured ε below the claim", GREEN, "ship ✓"),
                     ("buggy implementation: measured ε above the claim", RED, "block ✗")]
        for k, (txt, col, verdict) in enumerate(lines):
            t = ph(f, 40 + 10 * k, 10)
            y = 3.9 - k * 1.5
            ax.text(0.5, y, typed(txt, t), color=TEXT, fontsize=11, family="DejaVu Sans Mono")
            if t > 0.99:
                ax.text(9.5, y, verdict, color=col, fontsize=13, weight="bold", ha="right")
        caption(aC, "Our run", "We planted a real bug (noise added to the mean, not the sum). The accountant can't see it. The audit can.")

    scenes = [scene_keyboard, scene_hospital, scene_audit]

    def update(f):
        blank(ax, aC)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        k, local = divmod(f, scene_len)
        scenes[k](local)
        for i in range(len(scenes)):
            ax.add_patch(Circle((4.7 + i * 0.3, 0.3), 0.07, color=SKY if i == k else EDGE))

    save(fig, update, range(scene_len * len(scenes)), "real_world_use_cases.gif", fps=12)


ALL = dict(mechanism=mechanism, attack=attack, canary=canary, usecases=usecases)

if __name__ == "__main__":
    for name in sys.argv[1:] or ALL:
        ALL[name]()
