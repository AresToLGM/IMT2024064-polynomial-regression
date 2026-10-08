"""Train the selected polynomial models and predict the supplied test rows."""

import argparse
import csv
import hashlib
import warnings
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parent
FINAL_MODELS = {
    1: {"degree": 5, "alpha": 0.01, "tol": 1e-4, "max_iter": 50000},
    2: {"degree": 10, "alpha": 0.001, "tol": 1e-7, "max_iter": 200000},
}


class SelectColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns=(0, 1, 2)):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[:, list(self.columns)]


class LegendreFeatures(BaseEstimator, TransformerMixin):
    """Alternative polynomial basis with the same total-degree constraint."""
    def __init__(self, degree=2):
        self.degree = degree

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        self.powers_ = PolynomialFeatures(self.degree, include_bias=False).fit(X).powers_
        self.n_output_features_ = len(self.powers_)
        return self

    def transform(self, X):
        values = np.ones((len(X), self.n_output_features_))
        vandermonde = np.polynomial.legendre.legvander(X, self.degree)
        for column in range(self.n_features_in_):
            values *= vandermonde[:, column, self.powers_[:, column]]
        return values


def load_data(problem, part, data_dir=None):
    features = [f"x{i}" for i in range(1, {1: 6, 2: 3}[problem] + 1)]
    directory = Path(data_dir or ROOT / "data")
    filename = f"IMT2024064_{part}_var{problem}.csv"
    path = directory / filename
    # Browser uploads can place the supplied CSVs beside the Python files.
    # Support both the packaged data/ layout and that flat layout.
    if not path.is_file() and directory.resolve() == (ROOT / "data").resolve():
        path = ROOT / filename
    labeled = part == "train"
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == features + (["y"] if labeled else [])
        rows = list(reader)
    X = np.array([[float(row[name]) for name in features] for row in rows])
    y = np.array([float(row["y"]) for row in rows]) if labeled else None
    assert X.shape == (1000, len(features)) and np.isfinite(X).all()
    assert y is None or np.isfinite(y).all()
    return X, y, rows, features


def make_model(problem, degree=5, alpha=0.01, kind="lasso", l1_ratio=1.0,
               columns=None, basis="monomial", tol=None, max_iter=None):
    tol = tol if tol is not None else (1e-4 if problem == 1 else 1e-5)
    max_iter = max_iter or 50000
    estimator = (LinearRegression() if kind == "ols" else Ridge(alpha=alpha)
                 if kind == "ridge" else ElasticNet(alpha=alpha, l1_ratio=l1_ratio,
                                                    tol=tol, max_iter=max_iter))
    steps = []
    if columns is not None or problem == 2:
        steps.append(("select", SelectColumns(tuple(columns if columns is not None else range(3)))))
    poly = LegendreFeatures(degree) if basis == "legendre" else PolynomialFeatures(degree, include_bias=False)
    return Pipeline(steps + [("poly", poly), ("scale", StandardScaler()), ("model", estimator)])


def predict(problem, data_dir, output_dir):
    X, y, _, features = load_data(problem, "train", data_dir)
    test_X, _, rows, _ = load_data(problem, "test", data_dir)
    model = make_model(problem, **FINAL_MODELS[problem])
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X, y)
        values = model.predict(test_X)
    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
        raise RuntimeError("The final model did not converge")
    assert model[-1].dual_gap_ <= FINAL_MODELS[problem]["tol"] * np.var(y)
    assert np.isfinite(values).all()
    # Preserve the original implementation's duplicate-key convention.
    duplicates = {}
    for index, row in enumerate(rows):
        key = tuple(row[name] for name in features) if problem == 1 else tuple(test_X[index])
        if key in duplicates:
            values[index] = duplicates[key]
        else:
            duplicates[key] = values[index]
    destination = Path(output_dir) / f"IMT2024064_pred_var{problem}.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=features + ["y"])
        writer.writeheader()
        writer.writerows({**row, "y": repr(float(value))} for row, value in zip(rows, values))
    with destination.open(newline="", encoding="utf-8") as handle:
        saved = list(csv.DictReader(handle))
    assert len(saved) == 1000
    assert all(all(a[name] == b[name] for name in features) for a, b in zip(rows, saved))
    print(f"var{problem}: degree {FINAL_MODELS[problem]['degree']}, alpha {FINAL_MODELS[problem]['alpha']}; "
          f"1,000 predictions -> {destination}", flush=True)
    print(f"SHA256: {hashlib.sha256(destination.read_bytes()).hexdigest()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=["both", "1", "2"], default="both")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "predictions")
    args = parser.parse_args()
    for problem in ([1, 2] if args.problem == "both" else [int(args.problem)]):
        predict(problem, args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
