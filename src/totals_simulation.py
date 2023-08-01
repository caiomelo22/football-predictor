import json
import os
import warnings

import joblib
import pandas as pd

from utils import regression_cols
from utils import regression_functions as pf
from utils import regression_stats

warnings.filterwarnings("ignore")
print("Setup Complete")

save_model = False
league = "serie-a"
seasons = "2019-2024"
season_test = 2021
betting_starts_after_n_games = 0

score_cols = ["home_score", "away_score"]


def get_outcome(row):
    if row["Actual home_score"] > row["Actual away_score"]:
        return "H"
    elif row["Actual home_score"] < row["Actual away_score"]:
        return "A"
    else:
        return "D"


predictions = []
models = {}
for col in score_cols:
    # Read the data
    X_train, y_train, X_test, y_test = pf.get_league_data(
        league, seasons, season_test, col
    )

    df_predictions, best_model = pf.regression_model_evaluation(
        X_train, y_train, X_test, y_test, col
    )
    models[col] = best_model
    predictions.append(df_predictions)

df_predictions = pd.concat(predictions, axis=1)
df_predictions[f"Rounded Pred Score"] = (
    df_predictions["Pred home_score"] + df_predictions["Pred away_score"]
).round(0)
df_predictions[f"Score Difference"] = (
    df_predictions["Pred home_score"] + df_predictions["Pred away_score"]
) - (df_predictions["Actual home_score"] + df_predictions["Actual away_score"])
df_predictions[f"Score Difference Rounded"] = df_predictions[f"Score Difference"].round(
    0
)
df_predictions["Actual Outcome"] = df_predictions.apply(get_outcome, axis=1)
print("\nFinal predictions df:")
print(df_predictions)


def get_difference_stats(df_predictions):
    diff_cols = ["home_score", "away_score", "Score"]
    for col in diff_cols:
        diff_col = f"{col} Difference"
        diff_values = df_predictions[diff_col]

        print(
            f"\n{diff_col} Less than 0.5 diff: {len(diff_values[abs(diff_values) <= 0.5])/len(diff_values)}"
        )

        pred_col = f"Pred {col}"
        if pred_col in df_predictions.columns:
            pred_values = df_predictions[pred_col]
            print(f"{pred_col} Mean pred value: {pred_values.mean()}")

        actual_col = f"Actual {col}"
        if actual_col in df_predictions.columns:
            actual_values = df_predictions[actual_col]
            print(f"{actual_col} Mean actual value: {actual_values.mean()}")

    rounded_pred_values = df_predictions["Score Difference Rounded"]
    pred_higher_than_actual = len(rounded_pred_values[rounded_pred_values > 0]) / len(
        rounded_pred_values
    )
    print(
        f"{diff_col} Predicted value higher than actual: {round(pred_higher_than_actual, 2)}"
    )
    print(
        f"{diff_col} Predicted value lower than actual: {round(1-pred_higher_than_actual, 2)}"
    )
    print(f"{df_predictions['Rounded Pred Score'].value_counts()}")


get_difference_stats(df_predictions)

if save_model:
    cols_info = {"filtered_cols": regression_cols, "selected_stats": regression_stats}
    path = f"leagues/{league}/official/regression"
    if not os.path.exists(path):
        os.makedirs(path)

    with open(f"{path}/columns.json", "w") as json_file:
        json.dump(cols_info, json_file)

    for model in models.keys():
        joblib.dump(models[model], f"{path}/{model}.joblib")
