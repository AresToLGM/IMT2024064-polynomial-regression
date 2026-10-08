"""Reproduce the development split, staged searches, and validation checks.

Searches write optional logs to analysis_results/. They do not change the
frozen models or submission predictions in train_predict.py.
"""

import argparse
import json
import random
import warnings
from collections import defaultdict
from itertools import combinations

import numpy as np
from joblib import Parallel, delayed, parallel_backend
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold, cross_validate
from sklearn.preprocessing import PolynomialFeatures
from threadpoolctl import threadpool_limits

from train_predict import FINAL_MODELS, ROOT, LegendreFeatures, load_data, make_model


def split_indices(X, problem):
    """Reconstruct the original seed-42 split, preserving source row order."""
    counts = np.isin(X, (-1.0, 1.0)).sum(axis=1)
    labels = (np.where(counts == 0, 0, np.where(counts <= 3, 1, 2)) if problem == 1
              else np.minimum(counts, 2))
    groups = {group: np.flatnonzero(labels == group).tolist() for group in range(3)}
    quota = {group: len(groups[group]) * 200 for group in groups}
    target = {group: quota[group] // 1000 for group in groups}
    for group in sorted(groups, key=lambda g: -(quota[g] % 1000))[:200 - sum(target.values())]:
        target[group] += 1
    rng, reserved = random.Random(42), set()
    for group, indices in groups.items():
        if problem == 1:
            shuffled = indices.copy()
            rng.shuffle(shuffled)
            reserved.update(shuffled[:target[group]])
        else:
            sites = defaultdict(list)
            for index in indices:
                sites[tuple(X[index])].append(index)
            keys = list(sites)
            rng.shuffle(keys)
            reachable = {0: ()}
            for key in keys:
                size = len(sites[key])
                for total, selected in list(reachable.items()):
                    new_total = total + size
                    if new_total <= target[group] and new_total not in reachable:
                        reachable[new_total] = selected + (key,)
                if target[group] in reachable:
                    break
            for key in reachable[target[group]]:
                reserved.update(sites[key])
    development = np.array([i for i in range(1000) if i not in reserved])
    holdout = np.array([i for i in range(1000) if i in reserved])
    assert len(development) == 800 and len(holdout) == 200
    if problem == 2:
        assert not set(map(tuple, X[development])) & set(map(tuple, X[holdout]))
    return development, holdout


def folds(X, problem, seed):
    if problem == 1:
        return list(KFold(5, shuffle=True, random_state=seed).split(X))
    groups = np.unique(X, axis=0, return_inverse=True)[1]
    result = list(GroupKFold(5, shuffle=True, random_state=seed).split(X, groups=groups))
    assert all(not set(groups[a]) & set(groups[b]) for a, b in result)
    return result


def converged(estimator, X, y):
    fitted = estimator[-1]
    return float(not isinstance(fitted, ElasticNet) or fitted.n_iter_ < fitted.max_iter)


def choose_converged(results):
    indices = np.flatnonzero(results["mean_test_converged"] == 1)
    if not len(indices):
        raise RuntimeError("No candidate converged in every fold")
    return int(indices[np.argmax(results["mean_test_mse"][indices])])


def search_grid(problem, stage):
    n = {1: 6, 2: 3}[problem]
    all_columns = [tuple(range(n))]
    small = [subset for size in range(1, 4 if problem == 1 else 3)
             for subset in combinations(range(n), size)]
    sparse = ElasticNet(max_iter=50000, tol=1e-4 if problem == 1 else 1e-5)
    if stage == "initial":
        if problem == 1:
            large = [subset for size in range(4, 7) for subset in combinations(range(6), size)]
            return [{"select__columns": subsets, "poly__degree": degrees,
                     "model": [Ridge()], "model__alpha": [0.001, 0.1, 10.0]}
                    for subsets, degrees in ((small, list(range(1, 11))), (large, list(range(1, 6))))]
        return [
            {"select__columns": small, "poly__degree": list(range(1, 21)), "model": [LinearRegression()]},
            {"select__columns": all_columns, "poly__degree": list(range(1, 13)), "model": [LinearRegression()]},
            {"select__columns": small + all_columns, "poly__degree": list(range(1, 21)), "model": [Ridge()], "model__alpha": [0.0001, 0.01, 1.0, 100.0]},
        ]
    if stage == "refine":
        ridge = {"select__columns": all_columns, "poly__degree": list(range(3, 11)) if problem == 1 else list(range(8, 14)),
                 "model": [Ridge()], "model__alpha": [1, 3, 10, 30, 100, 300, 1000] if problem == 1 else [0.03, 0.1, 0.3, 1, 3, 10, 30]}
        lasso = {"select__columns": all_columns, "poly__degree": [3, 4, 5, 6] if problem == 1 else list(range(6, 13)),
                 "model": [sparse], "model__alpha": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3], "model__l1_ratio": [0.5, 0.8, 1.0]}
        return [ridge, lasso]
    if stage == "local":
        return [{"select__columns": all_columns, "poly__degree": [5] if problem == 1 else [11, 12, 13],
                 "model": [ElasticNet(l1_ratio=1, max_iter=150000, tol=1e-4 if problem == 1 else 1e-5)],
                 "model__alpha": [0.0075, 0.008, 0.009, 0.01, 0.011, 0.012, 0.013, 0.014, 0.015] if problem == 1 else [0.0006, 0.0008, 0.001, 0.0013, 0.0016, 0.002]}]
    if stage == "extend" and problem == 2:
        degrees = [6, 8, 10, 12, 14, 16, 18, 20]
        return [
            {"select__columns": all_columns, "poly": [LegendreFeatures()], "poly__degree": [6, 8, 10, 12], "model": [LinearRegression()]},
            {"select__columns": all_columns, "poly": [LegendreFeatures()], "poly__degree": degrees, "model": [Ridge()], "model__alpha": [0.01, 0.1, 1, 10, 100]},
            {"select__columns": all_columns, "poly": [LegendreFeatures()], "poly__degree": degrees, "model": [ElasticNet(l1_ratio=1, max_iter=50000, tol=1e-5)], "model__alpha": [0.003, 0.01, 0.03, 0.1]},
            {"select__columns": all_columns, "poly": [PolynomialFeatures(include_bias=False)], "poly__degree": [12, 14, 16, 20], "model": [ElasticNet(l1_ratio=1, max_iter=50000, tol=1e-5)], "model__alpha": [0.001, 0.003]},
            {"select__columns": all_columns, "poly__degree": [12], "model": [ElasticNet(l1_ratio=1, max_iter=50000, tol=1e-5)], "model__alpha": [0.0003]},
        ]
    raise ValueError("The extend stage applies to var2 only")


def grid_search(X, y, problem, stage, jobs):
    cv = folds(X, problem, 42)
    search = GridSearchCV(make_model(problem, columns=tuple(range(X.shape[1]))),
                          search_grid(problem, stage), cv=cv, n_jobs=jobs, pre_dispatch=jobs,
                          scoring={"mse": "neg_mean_squared_error", "r2": "r2", "converged": converged},
                          refit=choose_converged, error_score="raise", verbose=1)
    with threadpool_limits(limits=1), parallel_backend("loky", inner_max_num_threads=1):
        search.fit(X, y)
    result = search.cv_results_
    rows = []
    for index in np.argsort(-result["mean_test_mse"]):
        params = result["params"][index]
        estimator = params["model"]
        ratio = float(params.get("model__l1_ratio", getattr(estimator, "l1_ratio", 1.0)))
        kind = "ols" if isinstance(estimator, LinearRegression) else "ridge" if isinstance(estimator, Ridge) else "lasso" if ratio == 1 else "elasticnet"
        rows.append({"columns": list(params["select__columns"]), "degree": int(params["poly__degree"]),
                     "kind": kind, "alpha": params.get("model__alpha"), "l1_ratio": ratio,
                     "basis": "legendre" if isinstance(params.get("poly"), LegendreFeatures) else "monomial",
                     "mean_cv_mse": float(-result["mean_test_mse"][index]),
                     "mean_cv_r2": float(result["mean_test_r2"][index]),
                     "all_folds_converged": bool(result["mean_test_converged"][index] == 1)})
    output = {"holdout_used": False, "seed": 42, "candidates": rows,
              "best": next(row for row in rows if row["all_folds_converged"])}
    if stage == "initial":
        output["baselines"] = {}
        for name, model in (("training_mean", DummyRegressor()), ("linear", LinearRegression())):
            with threadpool_limits(limits=1):
                scores = cross_validate(model, X, y, cv=cv, scoring="neg_mean_squared_error")
            output["baselines"][name] = float(-scores["test_score"].mean())
    return output


def endpoint_masks(X, problem):
    count = np.isin(X, (-1.0, 1.0)).sum(axis=1)
    return ({"0-2": count <= 2, "3": count == 3, "4-6": count >= 4} if problem == 1
            else {"0": count == 0, "1": count == 1, "2-3": count >= 2})


def fit_fold(X, y, train, valid, problem, config, shares, strength):
    model = make_model(problem, **config, tol=1e-4 if problem == 1 else 1e-5,
                       max_iter=50000 if problem == 1 else 150000)
    params = {}
    if strength:
        weights = np.ones(len(train))
        for name, mask in endpoint_masks(X[train], problem).items():
            if mask.any():
                weights[mask] = (shares[name] / mask.mean()) ** strength
        params["model__sample_weight"] = weights / weights.mean()
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X[train], y[train], **params)
        predicted = model.predict(X[valid])
    warning_count = sum(issubclass(item.category, ConvergenceWarning) for item in caught)
    return valid, predicted, warning_count


def repeated_cv(X, y, test_X, problem, configs, seeds, jobs, strengths=None):
    masks = endpoint_masks(X, problem)
    shares = {name: float(mask.mean()) for name, mask in endpoint_masks(test_X, problem).items()}
    strengths = strengths or [0] * len(configs)
    repeats = [[] for _ in configs]
    with parallel_backend("loky", inner_max_num_threads=1):
        for seed in seeds:
            cv = folds(X, problem, seed)
            for index, (config, strength) in enumerate(zip(configs, strengths)):
                fitted = Parallel(n_jobs=jobs, pre_dispatch=jobs)(
                    delayed(fit_fold)(X, y, train, valid, problem, config, shares, strength) for train, valid in cv)
                values, coverage, warnings_count = np.full(len(y), np.nan), np.zeros(len(y)), 0
                for valid, predictions, warning_count in fitted:
                    values[valid], coverage[valid] = predictions, coverage[valid] + 1
                    warnings_count += warning_count
                assert np.all(coverage == 1) and np.isfinite(values).all()
                error = (values - y) ** 2
                proxy = sum(shares[name] * float(error[mask].mean()) for name, mask in masks.items())
                repeats[index].append({"seed": seed, "oof_mse": float(error.mean()), "oof_r2": float(r2_score(y, values)),
                                       "test_mix_proxy_mse": proxy, "convergence_warnings": warnings_count})
            print(f"var{problem}: repeat seed {seed} complete", flush=True)
    return {"holdout_used": False, "test_inputs_used": "Endpoint proportions only, for a descriptive proxy",
            "note": "Repeats reuse development rows. These are not independent test scores.",
            "results": [{"config": config, "weight_strength": strength,
                         "mean_oof_mse": float(np.mean([row["oof_mse"] for row in rows])),
                         "mean_test_mix_proxy_mse": float(np.mean([row["test_mix_proxy_mse"] for row in rows])),
                         "convergence_warnings": sum(row["convergence_warnings"] for row in rows), "repeats": rows}
                        for config, strength, rows in zip(configs, strengths, repeats)]}


def finalists(problem):
    if problem == 1:
        return [{"degree": 5, "kind": "ridge", "alpha": alpha} for alpha in (10, 30)] + [
            {"degree": 5, "alpha": alpha} for alpha in (0.0075, 0.01, 0.015)]
    return [{"degree": 10, "alpha": 0.001}, {"degree": 12, "alpha": 0.001},
            {"degree": 13, "alpha": 0.0013}, {"degree": 12, "kind": "ridge", "alpha": 3},
            {"degree": 12, "alpha": 0.001, "l1_ratio": 0.8}]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", type=int, choices=[1, 2], required=True)
    parser.add_argument("--stage", choices=["initial", "refine", "extend", "local", "repeat", "weights", "alpha-check", "holdout"], required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    assert args.jobs > 0 and args.repeats > 0
    X, y, _, _ = load_data(args.problem, "train")
    development, holdout = split_indices(X, args.problem)
    dev_X, dev_y = X[development], y[development]
    if args.stage in ("initial", "refine", "extend", "local"):
        output = grid_search(dev_X, dev_y, args.problem, args.stage, args.jobs)
    elif args.stage == "holdout":
        model = make_model(args.problem, **FINAL_MODELS[args.problem])
        with threadpool_limits(limits=1):
            model.fit(dev_X, dev_y)
            values = model.predict(X[holdout])
        output = {"holdout_mse": float(mean_squared_error(y[holdout], values)),
                  "holdout_r2": float(r2_score(y[holdout], values)),
                  "note": "Diagnostic comparison on the previously reused holdout; not a fresh test estimate."}
    else:
        test_X, _, _, _ = load_data(args.problem, "test")
        if args.stage == "weights":
            assert args.problem == 1, "Boundary weighting was investigated for var1"
            configs = [{"degree": 5, "alpha": 0.01}] * 5
            output = repeated_cv(dev_X, dev_y, test_X, 1, configs, range(200, 200 + args.repeats), args.jobs, [0, 0.25, 0.5, 0.75, 1])
        elif args.stage == "alpha-check":
            assert args.problem == 1, "This confirmation stage applies to var1"
            configs = [{"degree": 5, "alpha": alpha} for alpha in (0.0075, 0.008, 0.009, 0.01, 0.011, 0.012, 0.013, 0.014, 0.015)]
            screening = repeated_cv(dev_X, dev_y, test_X, 1, configs, range(600, 610), args.jobs)
            candidate = min(screening["results"], key=lambda row: row["mean_test_mix_proxy_mse"])["config"]
            confirmation = repeated_cv(dev_X, dev_y, test_X, 1, [{"degree": 5, "alpha": 0.01}, candidate], range(700, 720), args.jobs)
            current, alternative = confirmation["results"]
            gain = 1 - alternative["mean_test_mix_proxy_mse"] / current["mean_test_mix_proxy_mse"]
            wins = sum(b["test_mix_proxy_mse"] < a["test_mix_proxy_mse"] for a, b in zip(current["repeats"], alternative["repeats"]))
            output = {"screening": screening, "confirmation": confirmation, "relative_proxy_gain": gain,
                      "wins_out_of_20": wins, "practical_rule_met": gain >= 0.01 and wins >= 16,
                      "note": "Practical comparison criterion, not a statistical significance test."}
        else:
            output = repeated_cv(dev_X, dev_y, test_X, args.problem, finalists(args.problem), range(100, 100 + args.repeats), args.jobs)
    destination = ROOT / "analysis_results" / f"var{args.problem}_{args.stage}.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Saved {destination}", flush=True)
    if "best" in output:
        print(json.dumps(output["best"], indent=2))


if __name__ == "__main__":
    main()
