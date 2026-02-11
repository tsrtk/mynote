"""
Generate reference plots for logits/sigmoid/softmax and save them as PNG.

Run:
  python scripts/generate_plots.py

Outputs:
  plots/*.png
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # no GUI backend
import matplotlib.pyplot as plt


# ----------------------------
# math helpers
# ----------------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def softmax(z, axis=-1):
    z = np.array(z)
    zmax = np.max(z, axis=axis, keepdims=True)
    e = np.exp(z - zmax)
    return e / np.sum(e, axis=axis, keepdims=True)


# ----------------------------
# paths
# ----------------------------
root = Path(__file__).resolve().parent
plots_dir = root / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)

# ----------------------------
# 1) Sigmoid and Logit
# ----------------------------
x = np.linspace(-10, 10, 1000)
plt.figure(figsize=(6, 4))
plt.plot(x, sigmoid(x), label="sigmoid(x)")
plt.axhline(0.5, color="k", lw=0.7, alpha=0.4)
plt.axvline(0.0, color="k", lw=0.7, alpha=0.4)
plt.ylim(-0.05, 1.05)
plt.title("Sigmoid")
plt.xlabel("x (logit)")
plt.ylabel("p")
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(plots_dir / "sigmoid.png", dpi=160)
plt.close()

p = np.linspace(1e-4, 1 - 1e-4, 1000)
plt.figure(figsize=(6, 4))
plt.plot(p, logit(p), label="logit(p)=log(p/(1-p))", color="#d55e00")
plt.axhline(0.0, color="k", lw=0.7, alpha=0.4)
plt.axvline(0.5, color="k", lw=0.7, alpha=0.4)
plt.title("Logit (inverse of sigmoid)")
plt.xlabel("p")
plt.ylabel("x (log-odds)")
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(plots_dir / "logit.png", dpi=160)
plt.close()

# ----------------------------
# 2) Softmax (3-class) with a 1D sweep
# ----------------------------
t = np.linspace(-6, 6, 400)

# logits = [t, 0, -1]
Z1 = np.stack([t, np.zeros_like(t), -1 * np.ones_like(t)], axis=1)
S1 = softmax(Z1, axis=1)

plt.figure(figsize=(7, 4.2))
plt.plot(t, S1[:, 0], label="class0", lw=2)
plt.plot(t, S1[:, 1], label="class1", lw=2)
plt.plot(t, S1[:, 2], label="class2 (bias=-1)", lw=2)
plt.title("Softmax probabilities when logits = [t, 0, -1]")
plt.xlabel("t (logit for class0)")
plt.ylabel("probability")
plt.ylim(-0.02, 1.02)
plt.grid(True, alpha=0.25)
plt.legend(loc="best", fontsize=9)
plt.tight_layout()
plt.savefig(plots_dir / "softmax_3class.png", dpi=160)
plt.close()

# logits = [t, 0, 0]
Z2 = np.stack([t, np.zeros_like(t), np.zeros_like(t)], axis=1)
S2 = softmax(Z2, axis=1)

plt.figure(figsize=(7, 4.2))
plt.plot(t, S2[:, 0], label="class0", lw=2)
plt.plot(t, S2[:, 1], label="class1", lw=2)
plt.plot(t, S2[:, 2], label="class2", lw=2)
plt.title("Softmax probabilities when logits = [t, 0, 0]")
plt.xlabel("t (logit for class0)")
plt.ylabel("probability")
plt.ylim(-0.02, 1.02)
plt.grid(True, alpha=0.25)
plt.legend(loc="best", fontsize=9)
plt.tight_layout()
plt.savefig(plots_dir / "softmax_3class_symmetric.png", dpi=160)
plt.close()

# ----------------------------
# 3) Binary cross-entropy vs logit (stable form)
#   y=1: log(1+exp(-x))
#   y=0: log(1+exp( x))
# ----------------------------
x = np.linspace(-12, 12, 800)
ce_y1 = np.log1p(np.exp(-x))
ce_y0 = np.log1p(np.exp(x))

plt.figure(figsize=(7, 4.2))
plt.plot(x, ce_y1, label="y=1: log(1+exp(-x))", lw=2)
plt.plot(x, ce_y0, label="y=0: log(1+exp(x))", lw=2)
plt.title("Binary cross-entropy as a function of logit")
plt.xlabel("logit x")
plt.ylabel("loss")
plt.grid(True, alpha=0.25)
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig(plots_dir / "binary_ce_vs_logit.png", dpi=160)
plt.close()

print("Saved plots to:", plots_dir)
