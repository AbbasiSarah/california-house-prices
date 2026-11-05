"""Train and evaluate house-price models; writes results/ and models/.

    python train.py      # ~2 minutes
"""
import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict, train_test_split

sys.path.insert(0, str(Path(__file__).parent / "src"))
from housing import CAP, NUMERIC, CATEGORICAL, TARGET, load_data, models, spatial_groups  # noqa: E402

FIG = Path("results/figures")
ACC, GREY = "#2E7D32", "#B0BEC5"


def scores(y, p):
    return {"rmse": float(np.sqrt(mean_squared_error(y, p))), "mae": float(mean_absolute_error(y, p)),
            "r2": float(r2_score(y, p))}


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    df = load_data()
    X, y = df[NUMERIC + CATEGORICAL], df[TARGET].values
    Xtr, Xte, ytr, yte, gtr, _ = train_test_split(X, y, spatial_groups(df), test_size=0.2, random_state=42)
    print(f"{len(df)} districts; train {len(Xtr)}, test {len(Xte)}")

    res, test_pred = {}, {}
    for name, m in models().items():
        cv_random = cross_val_predict(m, Xtr, ytr, cv=KFold(5, shuffle=True, random_state=0), n_jobs=1)
        cv_spatial = cross_val_predict(m, Xtr, ytr, cv=GroupKFold(5), groups=gtr, n_jobs=1)
        m.fit(Xtr, ytr)
        p = m.predict(Xte)
        test_pred[name] = p
        res[name] = {"cv_random": scores(ytr, cv_random), "cv_spatial": scores(ytr, cv_spatial),
                     "test": scores(yte, p),
                     "test_uncapped_only": scores(yte[yte < CAP], p[yte < CAP])}
        print(f"{name:25s} test RMSE ${res[name]['test']['rmse']:,.0f}  MAE ${res[name]['test']['mae']:,.0f}  "
              f"R2 {res[name]['test']['r2']:.3f} | unseen-area CV R2 {res[name]['cv_spatial']['r2']:.3f}")

    best = min((n for n in res if not n.startswith("Baseline")), key=lambda n: res[n]["cv_random"]["rmse"])
    final = models()[best].fit(Xtr, ytr)
    Path("models").mkdir(exist_ok=True)
    pickle.dump(final, open("models/house_price_model.pkl", "wb"))

    imp = permutation_importance(final, Xte, yte, n_repeats=5, random_state=0, scoring="neg_root_mean_squared_error")
    importance = sorted(zip(X.columns, imp.importances_mean), key=lambda t: -t[1])
    p = test_pred[best]
    within = float(np.mean(np.abs(p - yte) / yte <= 0.20))

    report = {"data": {"districts": len(df), "train": len(Xtr), "test": len(Xte), "capped_at_500001": int((y >= CAP).sum())},
              "chosen_model": best, "results": res, "share_within_20pct_test": within,
              "permutation_importance_rmse_increase": {k: float(v) for k, v in importance}}
    Path("results/metrics.json").write_text(json.dumps(report, indent=2))
    print("chosen:", best, f"| {within:.1%} of test districts within ±20%")

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    # predicted vs actual
    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    ax.scatter(yte / 1e3, p / 1e3, s=4, alpha=0.3, color=ACC)
    ax.plot([0, 520], [0, 520], color="k", lw=0.8, ls=":")
    ax.axvline(CAP / 1e3, color=GREY, lw=1)
    ax.text(CAP / 1e3 - 8, 30, "census cap\n$500k", ha="right", fontsize=9, color="#607D8B")
    ax.set(xlabel="Actual median value ($1,000)", ylabel="Predicted ($1,000)",
           title=f"{best}: test set (R² {res[best]['test']['r2']:.2f})", xlim=(0, 520), ylim=(0, 620))
    fig.tight_layout(); fig.savefig(FIG / "predicted_vs_actual.png", dpi=150); plt.close(fig)
    # importance
    names, vals = zip(*importance[:9])
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.barh(list(names)[::-1], np.array(vals[::-1]) / 1e3, color=[GREY] * 8 + [ACC])
    ax.set(xlabel="Error increase when shuffled (RMSE, $1,000)", title="What drives the price prediction")
    fig.tight_layout(); fig.savefig(FIG / "feature_importance.png", dpi=150); plt.close(fig)
    # map of errors
    te = df.loc[Xte.index]
    err = (p - yte) / 1e3
    fig, ax = plt.subplots(figsize=(5.6, 6))
    sc = ax.scatter(te.longitude, te.latitude, c=np.clip(err, -150, 150), cmap="RdBu_r", s=5, vmin=-150, vmax=150)
    fig.colorbar(sc, ax=ax, shrink=0.7, label="Prediction − actual ($1,000)")
    ax.set(xlabel="Longitude", ylabel="Latitude", title="Where the model over- and under-prices")
    ax.set_aspect(1.15)
    fig.tight_layout(); fig.savefig(FIG / "error_map.png", dpi=150); plt.close(fig)
    # random vs spatial
    order = [n for n in res if not n.startswith("Baseline")]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    yy = np.arange(len(order))
    ax.barh(yy + 0.2, [res[n]["cv_random"]["r2"] for n in order], 0.38, color=ACC, label="Random split")
    ax.barh(yy - 0.2, [res[n]["cv_spatial"]["r2"] for n in order], 0.38, color=GREY, label="Unseen areas (spatial split)")
    for i, n in enumerate(order):
        ax.text(res[n]["cv_random"]["r2"] + 0.01, i + 0.2, f"{res[n]['cv_random']['r2']:.2f}", va="center", fontsize=9)
        ax.text(res[n]["cv_spatial"]["r2"] + 0.01, i - 0.2, f"{res[n]['cv_spatial']['r2']:.2f}", va="center", fontsize=9)
    ax.set(yticks=yy, yticklabels=order, xlim=(0, 1), xlabel="R² (5-fold cross-validation)",
           title="Accuracy drops in areas the model has never seen")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=2, fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "random_vs_spatial.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
