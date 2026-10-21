# %% [markdown]
# # Tips multivariable linear regression
# Exportable notebook-style Python script. Compares holdout, k-fold, and
# leave-one-out validation, then logs evaluation and provenance to MLflow.

# %%
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    KFold,
    LeaveOneOut,
    ShuffleSplit,
    cross_validate,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


RANDOM_STATE = 42
TARGET = "tip"
NUMERIC_FEATURES = ["total_bill", "size"]
CATEGORICAL_FEATURES = ["sex", "smoker", "day", "time"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_pipeline() -> Pipeline:
	"""Preprocess numeric and categorical predictors and fit linear regression."""
	preprocessing = ColumnTransformer(
		[
			("numeric", "passthrough", NUMERIC_FEATURES),
			(
				"categorical",
				OneHotEncoder(handle_unknown="ignore"),
				CATEGORICAL_FEATURES,
			),
		]
	)
	return Pipeline(
		[("preprocessing", preprocessing), ("regressor", LinearRegression())]
	)


def provenance() -> dict:
	"""Return source hash and Git revision when available."""
	source = Path(__file__).resolve()
	try:
		revision = subprocess.run(
			["git", "rev-parse", "HEAD"],
			cwd=source.parent,
			check=True,
			capture_output=True,
			text=True,
		).stdout.strip()
	except (OSError, subprocess.CalledProcessError):
		revision = "unavailable"
	return {
		"source_file": source.name,
		"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
		"git_commit": revision,
	}


def regression_metrics(actual, predicted) -> dict:
	return {
		"r2": float(r2_score(actual, predicted)),
		"rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
		"mae": float(mean_absolute_error(actual, predicted)),
	}


def coefficient_records(model: Pipeline, validation_method: str, split: int) -> list:
	preprocessing = model.named_steps["preprocessing"]
	regressor = model.named_steps["regressor"]
	parameters = {
		"intercept": float(regressor.intercept_),
		**dict(
			zip(
				preprocessing.get_feature_names_out(),
				regressor.coef_,
				strict=True,
			)
		),
	}
	return [
		{
			"validation_method": validation_method,
			"split": split,
			"parameter": parameter,
			"value": float(value),
		}
		for parameter, value in parameters.items()
	]


def coefficient_summary(records: pd.DataFrame) -> tuple[dict, list]:
	metrics = {}
	summaries = []
	grouped = records.groupby(["validation_method", "parameter"])["value"]
	for (method, parameter), values in grouped:
		stats = values.agg(["mean", "std", "var"])
		summary = {
			"validation_method": method,
			"parameter": parameter,
			"count": int(values.count()),
			"mean": float(stats["mean"]),
			"std": None if pd.isna(stats["std"]) else float(stats["std"]),
			"variance": None if pd.isna(stats["var"]) else float(stats["var"]),
		}
		summaries.append(summary)
		metric_prefix = f"coefficient_{method}_{parameter}".replace("-", "_")
		metrics[f"{metric_prefix}_mean"] = summary["mean"]
		if summary["std"] is not None:
			metrics[f"{metric_prefix}_std"] = summary["std"]
			metrics[f"{metric_prefix}_variance"] = summary["variance"]
	return metrics, summaries


def main() -> None:
	# Dataset and deterministic fingerprint for reproducibility.
	tips = sns.load_dataset("tips").dropna(subset=FEATURES + [TARGET]).copy()
	X, y = tips[FEATURES], tips[TARGET]
	validation_pipeline = build_pipeline()


	data_hash = hashlib.sha256(
		pd.util.hash_pandas_object(tips, index=True).values.tobytes()
	).hexdigest()

	# Validation 1: single 80/20 train-test split.
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.2, random_state=RANDOM_STATE
	)
	holdout_model = build_pipeline().fit(X_train, y_train)
	holdout = regression_metrics(y_test, holdout_model.predict(X_test))
	metrics = {f"holdout_{key}": value for key, value in holdout.items()}
	coefficient_rows = coefficient_records(holdout_model, "holdout", 1)

	# Validation 2: shuffled five-fold cross-validation.
	scoring = {
		"r2": "r2",
		"rmse": "neg_root_mean_squared_error",
		"mae": "neg_mean_absolute_error",
	}
	kfold = cross_validate(
		build_pipeline(),
		X,
		y,
		cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
		scoring=scoring,
		return_estimator=True,
	)
	for name in scoring:
		values = kfold[f"test_{name}"]
		if name in ("rmse", "mae"):
			values = -values
		metrics[f"kfold_{name}_mean"] = float(values.mean())
		metrics[f"kfold_{name}_std"] = float(values.std())
	for split, estimator in enumerate(kfold["estimator"], start=1):
		coefficient_rows.extend(coefficient_records(estimator, "kfold", split))

	# Validation 3: leave-one-out cross-validation; aggregate out-of-fold errors.
	loo = cross_validate(
		build_pipeline(),
		X,
		y,
		cv=LeaveOneOut(),
		scoring={
			"rmse": "neg_root_mean_squared_error",
			"mae": "neg_mean_absolute_error",
		},
		n_jobs=-1,
		return_estimator=True,
	)
	metrics["loo_rmse_mean"] = float(-loo["test_rmse"].mean())
	metrics["loo_mae_mean"] = float(-loo["test_mae"].mean())
	for split, estimator in enumerate(loo["estimator"], start=1):
		coefficient_rows.extend(coefficient_records(estimator, "loo", split))
	coefficient_frame = pd.DataFrame(coefficient_rows)
	coefficient_metrics, coefficient_summaries = coefficient_summary(
		coefficient_frame
	)
	metrics.update(coefficient_metrics)

    # ------------------------------------------------------------
    # Additional validation comparison suite
    # ------------------------------------------------------------

	validation_strategies = {
		"holdout": ShuffleSplit(
			n_splits=1,
			test_size=0.2,
			random_state=RANDOM_STATE,
		),
		"kfold_5": KFold(
			n_splits=5,
			shuffle=True,
			random_state=RANDOM_STATE,
		),
		"shufflesplit_10": ShuffleSplit(
			n_splits=10,
			test_size=0.2,
			random_state=RANDOM_STATE,
		),
		"leave_one_out": LeaveOneOut(),
	}

	print("\n--- Comprehensive Validation Suite ---")

	for name, cv_strategy in validation_strategies.items():

		scores = cross_val_score(
			validation_pipeline,
			X,
			y,
			cv=cv_strategy,
			scoring="neg_mean_squared_error",
			n_jobs=-1,
		)

		mse_mean = -scores.mean()
		mse_std = scores.std()

		print(
			f"{name:25} | "
			f"Average MSE: {mse_mean:.4f} | "
			f"Std Dev: {mse_std:.4f}"
		)

		metrics[f"{name}_mse_mean"] = float(mse_mean)
		metrics[f"{name}_mse_std"] = float(mse_std)

	# ------------------------------------------------------------
	# Bootstrap validation
	# ------------------------------------------------------------

	n_bootstraps = 1000
	bootstrap_mses = []

	rng = np.random.default_rng(RANDOM_STATE)

	for i in range(n_bootstraps):

		train_idx = rng.choice(
			len(X),
			size=len(X),
			replace=True,
		)

		test_idx = np.setdiff1d(
			np.arange(len(X)),
			np.unique(train_idx),
		)

		if len(test_idx) == 0:
			continue

		X_train = X.iloc[train_idx]
		y_train = y.iloc[train_idx]

		X_test = X.iloc[test_idx]
		y_test = y.iloc[test_idx]

		bootstrap_model = build_pipeline()

		bootstrap_model.fit(X_train, y_train)

		y_pred = bootstrap_model.predict(X_test)

		mse = mean_squared_error(
			y_test,
			y_pred,
		)

		bootstrap_mses.append(mse)

	bootstrap_mean = np.mean(bootstrap_mses)
	bootstrap_std = np.std(bootstrap_mses)

	bootstrap_lower = np.percentile(
		bootstrap_mses,
		2.5,
	)

	bootstrap_upper = np.percentile(
		bootstrap_mses,
		97.5,
	)

	print(f"\nBootstrap Average MSE: {bootstrap_mean:.4f}")
	print(f"Bootstrap Std Dev: {bootstrap_std:.4f}")
	print(
		f"95% Bootstrap CI: "
		f"[{bootstrap_lower:.4f}, {bootstrap_upper:.4f}]"
	)

	metrics["bootstrap_mse_mean"] = float(bootstrap_mean)
	metrics["bootstrap_mse_std"] = float(bootstrap_std)
	metrics["bootstrap_ci_lower"] = float(bootstrap_lower)
	metrics["bootstrap_ci_upper"] = float(bootstrap_upper)

	code_info = provenance()

	mlflow.set_experiment("tips-multivariable-linear-regression")

	with mlflow.start_run() as run:

		mlflow.log_params(
			{
				"estimator": "LinearRegression",
				"target": TARGET,
				"features": ",".join(FEATURES),
				"numeric_features": ",".join(NUMERIC_FEATURES),
				"categorical_features": ",".join(CATEGORICAL_FEATURES),
				"holdout_test_size": 0.2,
				"random_state": RANDOM_STATE,
				"kfold_splits": 5,
				"validation_methods": "holdout,kfold,loo",
				"dataset_rows": len(tips),
			}
		)

		mlflow.log_metrics(metrics)

		mlflow.set_tags(
			{
				**code_info,
				"dataset": "seaborn.tips",
				"dataset_sha256": data_hash,
			}
		)

		mlflow.log_dict(
			{**code_info, "dataset": "seaborn.tips", "dataset_sha256": data_hash},
			"provenance.json",
		)

		mlflow.log_dict(
			coefficient_summaries,
			"coefficient_stability.json",
		)

		final_model = build_pipeline().fit(X, y)

		mlflow.sklearn.log_model(
			final_model,
			artifact_path="model",
		)

		with tempfile.TemporaryDirectory() as artifact_dir:

			coefficient_csv = (
				Path(artifact_dir)
				/ "validation_coefficients.csv"
			)

			coefficient_frame.to_csv(
				coefficient_csv,
				index=False,
			)

			mlflow.log_artifact(
				str(coefficient_csv)
			)

			bootstrap_csv = (
				Path(artifact_dir)
				/ "bootstrap_results.csv"
			)

			pd.DataFrame(
				{
					"bootstrap_mse": bootstrap_mses
				}
			).to_csv(
				bootstrap_csv,
				index=False,
			)

			mlflow.log_artifact(
				str(bootstrap_csv)
			)

			plot_path = (
				Path(artifact_dir)
				/ "coefficient_boxplot.png"
			)

			fig, axis = plt.subplots(
				figsize=(
					12,
					max(
						6,
						0.5 * coefficient_frame["parameter"].nunique(),
					),
				)
			)

			sns.boxplot(
				data=coefficient_frame,
				x="value",
				y="parameter",
				hue="validation_method",
				ax=axis,
			)

			axis.set(
				title="Regression coefficient stability by validation method"
			)

			axis.set(
				xlabel="Coefficient value",
				ylabel="Parameter",
			)

			fig.tight_layout()

			fig.savefig(
				plot_path,
				dpi=150,
			)

			plt.close(fig)

			mlflow.log_artifact(
				str(plot_path)
			)

			bootstrap_plot = (
				Path(artifact_dir)
				/ "bootstrap_mse_distribution.png"
			)

			fig, axis = plt.subplots(
				figsize=(8, 5)
			)

			axis.hist(
				bootstrap_mses,
				bins=30,
			)

			axis.axvline(
				bootstrap_lower,
				color="red",
				linestyle="--",
				label="2.5%",
			)

			axis.axvline(
				bootstrap_upper,
				color="red",
				linestyle="--",
				label="97.5%",
			)

			axis.set_title(
				"Bootstrap MSE Distribution"
			)

			axis.set_xlabel("MSE")
			axis.set_ylabel("Frequency")

			axis.legend()

			fig.tight_layout()

			fig.savefig(
				bootstrap_plot,
				dpi=150,
			)

			plt.close(fig)

			mlflow.log_artifact(
				str(bootstrap_plot)
			)

		print(
			json.dumps(
				{
					"run_id": run.info.run_id,
					"metrics": metrics,
				},
				indent=2,
			)
		)

if __name__ == "__main__":
	main()
