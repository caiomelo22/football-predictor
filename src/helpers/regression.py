import numpy as np
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    VotingRegressor,
)
from sklearn.linear_model import Ridge, LinearRegression, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error

from helpers.classification import build_pipeline
    
def get_regression_models(random_state: int, voting_models: list[str] | None = None) -> dict:
    base_estimators = {
        "rf_regressor": RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            random_state=random_state,
            n_jobs=-1,
        ),
        "ridge": Ridge(alpha=1.0, random_state=random_state),
        "linear_regression": LinearRegression(n_jobs=-1),
        "lasso": Lasso(alpha=0.01, random_state=random_state, max_iter=5000),
        "knn_regressor": KNeighborsRegressor(
            n_neighbors=10,
            weights="distance",
            n_jobs=-1,
        ),
        "decision_tree": DecisionTreeRegressor(max_depth=10, random_state=random_state),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=random_state,
        ),
        "hist_gb": HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            max_depth=None,
            random_state=random_state,
        ),
        "mlp_regressor": MLPRegressor(
            hidden_layer_sizes=(100, 50),
            activation="relu",
            solver="adam",
            max_iter=500,
            random_state=random_state,
        ),
        "svr": SVR(kernel="rbf", C=1.0, epsilon=0.1),
    }

    # Wrap automatically for multi-output support
    models_dict = {
        name: {
            "estimator": est,
            "params": None,
            "score": None,
        }
        for name, est in base_estimators.items()
    }

    # If a list of models is passed, add a VotingRegressor
    if voting_models:
        estimators = [
            (name, models_dict[name]["estimator"])
            for name in voting_models
            if name in models_dict
        ]

        if estimators:
            voting_regressor = VotingRegressor(estimators=estimators, n_jobs=-1)
            models_dict["voting_regressor"] = {
                "estimator": voting_regressor,
                "params": None,
                "score": None,
            }

    return models_dict

def simulate_with_regression(
    matches: pd.DataFrame,
    start_season,
    season,
    features,
    target_cols=["home_score", "away_score"],
    random_state=0,
    preprocess=True,
    voting_models=None,
):
    matches_filtered = matches[
        (matches["season"] >= start_season) & (matches["season"] <= season)
    ]
    matches_filtered = matches_filtered.dropna(subset=features + target_cols)

    train_set = matches_filtered[matches_filtered["season"] < season]
    test_set = matches_filtered[matches_filtered["season"] == season]

    X_train = train_set[features]
    X_test = test_set[features]

    y_train_dict = dict()
    y_test_dict = dict()

    for col in target_cols:
        y_train_dict[col] = train_set[col].values
        y_test_dict[col] = test_set[col].values

    models_dict = get_regression_models(random_state, voting_models)

    if not len(X_train):
        return matches, models_dict

    for model in models_dict.keys():
        for col in target_cols:
            my_pipeline = build_pipeline(X_train, models_dict[model]["estimator"], preprocess)

            y_train = y_train_dict[col]
            y_test = y_test_dict[col]

            # Preprocessing of training data, fit model
            my_pipeline.fit(X_train, y_train)

            if not len(X_test):
                continue
            
            y_pred = my_pipeline.predict(X_test)

            models_dict[model][f"score_{col}"] = my_pipeline.score(X_test, y_test)
            models_dict[model][f"pipeline_{col}"] = my_pipeline

            matches.loc[X_test.index, f"{col}_pred_{model}"] = y_pred

    return matches, models_dict

def plot_cumulative_profit(matches, selected_models, market, plot_threshold=0):
    plt.figure(figsize=(12, 6))
    for model in selected_models:
        cum_col = f"CumulativeProfit{market}_{model}"
        if matches[cum_col].iloc[-1] > plot_threshold:
            plt.plot(matches["date"], matches[cum_col], label=f"{market} {model}")
    plt.title(f"Cumulative Profit - {market}")
    plt.xlabel("Game")
    plt.ylabel("Cumulative Profit")
    plt.legend()
    plt.grid(True)
    plt.show()

def get_regression_model_scores(matches, model):
    metric_columns = [
        "home_score",
        "away_score",
        f"home_score_pred_{model}",
        f"away_score_pred_{model}",
    ]
    matches_filtered = matches.dropna(subset=metric_columns)

    home_true = matches_filtered["home_score"]
    away_true = matches_filtered["away_score"]
    
    home_pred = matches_filtered[f"home_score_pred_{model}"]
    away_pred = matches_filtered[f"away_score_pred_{model}"]
    
    home_r2 = r2_score(home_true, home_pred)
    away_r2 = r2_score(away_true, away_pred)
    
    home_mae = mean_absolute_error(home_true, home_pred)
    away_mae = mean_absolute_error(away_true, away_pred)
    
    return home_r2, away_r2, home_mae, away_mae

def show_regression_models_score(matches, selected_models):
    model_scores = {}
    
    for model in selected_models:
        home_r2, away_r2, home_mae, away_mae = get_regression_model_scores(matches, model)
        model_scores[model] = {
            "home_r2": home_r2,
            "away_r2": away_r2,
            "total_r2": (home_r2 + away_r2) / 2,
            "home_mae": home_mae,
            "away_mae": away_mae,
        }

    # Sort models by average R²
    sorted_models = sorted(model_scores.items(), key=lambda x: x[1]["total_r2"], reverse=True)

    for model, scores in sorted_models:
        home_r2 = scores["home_r2"]
        away_r2 = scores["away_r2"]
        home_mae = scores["home_mae"]
        away_mae = scores["away_mae"]
    
        print(f"\n{model} metrics:")
        print(f"Home score -> R²: {home_r2:.4f}, MAE: {home_mae:.4f}")
        print(f"Away score -> R²: {away_r2:.4f}, MAE: {away_mae:.4f}")

def profit_1x2_regression(row, model, default_value=1, min_odds=1.01):
    # Predict winner
    pred_home = row[f"home_score_pred_{model}"]
    pred_away = row[f"away_score_pred_{model}"]
    if pd.isna(pred_home) or pd.isna(pred_away):
        return 0
    if pred_home > pred_away:
        pred_res = "H"
        odds = row["home_odds"]
    elif pred_home < pred_away:
        pred_res = "A"
        odds = row["away_odds"]
    else:
        pred_res = "D"
        odds = row["draw_odds"]
    # Only bet if odds above threshold
    if odds < min_odds:
        return 0
    # Win/loss
    actual_res = "H" if row["home_score"] > row["away_score"] else ("A" if row["home_score"] < row["away_score"] else "D")
    if pred_res == actual_res:
        return odds * default_value - default_value
    else:
        return -default_value

def _settle_from_adjusted(adjusted, stake, odds, eps=1e-9):
    """Return profit given the adjusted margin for the selected side."""
    pnl_full = stake * (odds - 1)
    if adjusted > 0.5 + eps:            # Full win
        return pnl_full
    elif adjusted > 0 + eps:            # Half win
        return pnl_full / 2
    elif abs(adjusted) <= eps:          # Push
        return 0.0
    elif adjusted >= -0.5 - eps:        # Half loss
        return -stake / 2
    else:                               # Full loss
        return -stake


def profit_ahc_regression(row, model, default_value=1, min_odds=1.01):
    pred_home = row.get(f"home_score_pred_{model}")
    pred_away = row.get(f"away_score_pred_{model}")
    line = row.get("ahc_line")

    if pd.isna(pred_home) or pd.isna(pred_away) or pd.isna(line):
        return 0.0

    # Decide the side by comparing prediction to the handicap line (away-referenced)
    pred_margin_vs_line = (pred_home - pred_away) - line
    eps = 1e-9
    if abs(pred_margin_vs_line) <= eps:
        return 0.0  # No bet if predicted exactly on the line

    bet_on_home = pred_margin_vs_line > 0
    odds = row["home_ahc_odds"] if bet_on_home else row["away_ahc_odds"]
    if pd.isna(odds) or odds < min_odds:
        return 0.0

    # Actual margin
    actual_diff = row["home_score"] - row["away_score"]

    # Adjusted margin from the perspective of the selected side (line is AWAY-referenced)
    s = 1 if bet_on_home else -1
    adjusted = s * actual_diff - s * line

    return _settle_from_adjusted(adjusted, default_value, odds, eps=eps)


def profit_totals_regression(row, model, default_value=1, min_odds=1.01):
    pred_home = row.get(f"home_score_pred_{model}")
    pred_away = row.get(f"away_score_pred_{model}")
    line = row.get("totals_line")

    if pd.isna(pred_home) or pd.isna(pred_away) or pd.isna(line):
        return 0.0

    pred_total = pred_home + pred_away
    pred_margin = pred_total - line
    eps = 1e-9
    if abs(pred_margin) <= eps:
        return 0.0  # No bet if predicted exactly on the line

    bet_over = pred_margin > 0
    odds = row["overs_odds"] if bet_over else row["unders_odds"]
    if pd.isna(odds) or odds < min_odds:
        return 0.0

    actual_total = row["home_score"] + row["away_score"]

    # Adjusted margin from the selected side’s perspective
    adjusted = (actual_total - line) if bet_over else (line - actual_total)

    return _settle_from_adjusted(adjusted, default_value, odds, eps=eps)

def get_regression_simulation_results(
    matches,
    selected_models,
    plot_threshold=0,
    default_value=1,
    min_odds_1x2=1.01,
    min_odds_ahc=1.01,
    min_odds_totals=1.01,
):
    show_regression_models_score(matches, selected_models)

    # Calculate and plot profits
    for model in selected_models:
        matches[f"Profit1x2_{model}"] = matches.apply(lambda row: profit_1x2_regression(row, model, default_value, min_odds_1x2), axis=1)
        matches[f"CumulativeProfit1x2_{model}"] = matches[f"Profit1x2_{model}"].cumsum()
        
        matches[f"ProfitAHC_{model}"] = matches.apply(lambda row: profit_ahc_regression(row, model, default_value, min_odds_ahc), axis=1)
        matches[f"CumulativeProfitAHC_{model}"] = matches[f"ProfitAHC_{model}"].cumsum()
        
        matches[f"ProfitTotals_{model}"] = matches.apply(lambda row: profit_totals_regression(row, model, default_value, min_odds_totals), axis=1)
        matches[f"CumulativeProfitTotals_{model}"] = matches[f"ProfitTotals_{model}"].cumsum()

    # Plot all markets
    for market in ["1x2", "AHC", "Totals"]:
        print(f"\nCumulative Profit for {market} market:")

        plot_cumulative_profit(matches, selected_models, market, plot_threshold)

        # Create list of model results to sort
        model_results = []
        for model in selected_models:
            profit_col = f"CumulativeProfit{market}_{model}"
            bet_col = f"Profit{market}_{model}"
            cum_profit = round(matches.iloc[-1][profit_col], 4)
            num_bets = len(matches[matches[bet_col] != 0])
            avg_profit = round(cum_profit / num_bets, 4) if num_bets > 0 else 0
            model_results.append({
                'model': model,
                'cum_profit': cum_profit,
                'num_bets': num_bets,
                'avg_profit': avg_profit
            })

        # Sort by cumulative profit in descending order
        model_results.sort(key=lambda x: x['cum_profit'], reverse=True)

        # Print sorted results
        for result in model_results:
            print(f"{market} {result['model'].ljust(20)} --> ({str(result['cum_profit']).rjust(7)}/{str(result['num_bets']).rjust(3)}): {result['avg_profit']}")

    return matches

