import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import load
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
print("Setup Complete")


def separate_dataset_info(X):
    y = X.winner

    odds_cols = [
        "date",
        "season",
        "home_team",
        "away_team",
        "home_odds",
        "away_odds",
        "draw_odds",
    ]
    odds = X[odds_cols]

    for c in odds_cols:
        if "odds" in c:
            odds[c] = pd.to_numeric(odds[c], errors="coerce")
    odds.dropna(inplace=True)

    X.drop(
        ["winner", "home_score", "away_score", "home_odds", "away_odds", "draw_odds"],
        axis=1,
        inplace=True,
    )

    _, numerical_cols, _ = set_numerical_categorical_cols(X)
    return X[numerical_cols], y, odds


def get_league_data(league, seasons, season_test):
    # Read the data
    X_full = pd.read_csv(
        f"./leagues/{league}/formatted_data/{seasons}.csv", index_col=0
    )
    X_full.replace(" ", np.nan, inplace=True)
    X_full = X_full.dropna(subset=["home_odds", "away_odds", "draw_odds"], how="any")
    X_test_full = X_full[X_full["season"] == season_test]
    X_full = X_full[X_full["season"] < season_test]

    # Remove rows with missing target, separate target from predictors
    y = X_full.winner
    X_full.drop(
        ["winner", "home_score", "away_score", "home_odds", "away_odds", "draw_odds"],
        axis=1,
        inplace=True,
    )

    print(X_test_full[["date", "season", "home_team", "away_team", "winner"]])
    X_test_full, y_test, odds_test = separate_dataset_info(X_test_full)

    return X_full, y, X_test_full, y_test, odds_test


def set_numerical_categorical_cols(X: pd.DataFrame):
    # "Cardinality" means the number of unique values in a column
    # Select categorical columns with relatively low cardinality
    categorical_cols = [
        cname
        for cname in X.columns
        if X[cname].dtype == "object" and X[cname].nunique() < 10
    ]

    # Select numerical columns
    numerical_cols = [cname for cname in X.columns if X[cname].dtype == "float64"]

    # Select int columns
    int_cols = [cname for cname in X.columns if X[cname].dtype in ["int64", "int32"]]

    return categorical_cols, numerical_cols, int_cols


def filter_datasets(X_full, y, X_test_full, filtered_cols):
    y_train = y.copy()
    X_train = X_full[filtered_cols]
    X_test = X_test_full[filtered_cols]

    return X_train, y_train, X_test


def plot_feature_corr_chart(X, numerical_cols):
    plt.figure(figsize=(12, 12))
    plot_data = X[numerical_cols].corr()
    sns.heatmap(data=plot_data)
    plt.show()


def scale_dataset(df, scaler, just_transform=False):
    cols = df.columns
    if just_transform:
        df = pd.DataFrame(scaler.transform(df), columns=cols)
    else:
        df = pd.DataFrame(scaler.fit_transform(df), columns=cols)
    return df


def scale_test_values(X_test, scaler, features_to_explore):
    X_test_scaled = scale_dataset(
        X_test.loc[:, features_to_explore], scaler, just_transform=True
    )
    return X_test_scaled


def scale_train_values(X):
    # Standardize
    scaler = MinMaxScaler()
    X_scaled = scale_dataset(X, scaler)
    return X_scaled, scaler


def scale_values(X_train, X_test, features_to_explore):
    X_train_scaled, scaler = scale_train_values(X_train, features_to_explore)
    X_test_scaled = scale_test_values(X_test, scaler, features_to_explore)
    return X_train_scaled, X_test_scaled, scaler


def get_models():
    models_dict = {
        "naive_bayes": {
            "estimator": GaussianNB(var_smoothing=1e-7),
            "params": None,
            "score": None,
            "voting": False,
        },
        "knn": {
            "estimator": KNeighborsClassifier(n_neighbors=40),
            "params": None,
            "score": None,
            "voting": True,
        },
        "logistic_regression": {
            "estimator": LogisticRegression(random_state=0),
            "params": None,
            "score": None,
            "voting": False,
        },
        "svm": {
            "estimator": SVC(probability=True, random_state=0),
            "params": None,
            "score": None,
            "voting": False,
        },
        "random_forest": {
            "estimator": RandomForestClassifier(random_state=0, n_estimators=750),
            "params": None,
            "score": None,
            "voting": True,
        },
        "mlp": {
            "estimator": MLPClassifier(random_state=0),
            "params": None,
            "score": None,
            "voting": False,
        },
    }

    voting_classifier_estimators = []

    for model in models_dict.keys():
        if models_dict[model]["voting"]:
            voting_classifier_estimators.append(
                (model, models_dict[model]["estimator"])
            )

    if voting_classifier_estimators:
        models_dict["voting_classifier"] = {
            "estimator": VotingClassifier(
                estimators=voting_classifier_estimators, voting="soft"
            )
        }

    return models_dict


def build_pipeline(X_train, y_train, model, epochs=10, batch_size=32):
    categorical_cols, numerical_cols, int_cols = set_numerical_categorical_cols(X_train)

    # Preprocessing for numerical data
    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
        ]
    )

    # Preprocessing for int data
    int_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
        ]
    )

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ]
    )

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("int", int_transformer, int_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    # Bundle preprocessing and modeling code in a pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    # Preprocessing of training data, fit model
    pipeline.fit(X_train, y_train)

    return pipeline


def get_pred_odds(probs):
    return 1 / probs


def get_bet_value(odds, probs, bankroll, strategy="kelly"):
    unit_value = get_bet_unit_value(odds, probs, strategy)
    return unit_value * bankroll


def get_bet_odds_probs(bet):
    if bet["pred"] == "H":
        return bet["home_odds"], bet["home_probs"]
    if bet["pred"] == "A":
        return bet["away_odds"], bet["away_probs"]
    if bet["pred"] == "D":
        return bet["draw_odds"], bet["draw_probs"]


def bet_worth_it(bet_worth, odds):
    return bet_worth >= 5 and odds > 1.7


def get_bet_unit_value(odds, probs, strategy):
    if strategy == "kelly":
        q = 1 - probs  # Probability of losing
        b = odds - 1  # Net odds received on the bet (including the stake)
        kelly_fraction = ((probs * b - q) / b) * 0.5
        return round(
            min(kelly_fraction, 1.0), 4
        )  # Limit the bet to 100% of the bankroll
    elif strategy == "bankroll_pct":
        return 0.05


def get_bet_unit_value_by_row(row, strategy):
    odds, probs = get_bet_odds_probs(row)
    return get_bet_unit_value(odds, probs, strategy)


def get_bet_value_by_row(row, bankroll, strategy="kelly"):
    odds, probs = get_bet_odds_probs(row)
    return get_bet_value(odds, probs, bankroll, strategy)


def get_match_profit(row):
    odds, _ = get_bet_odds_probs(row)
    if not bet_worth_it(row["bet_worth"], odds):
        return 0
    if row["winner"] == row["pred"]:
        return (odds * row["bet_worth"]) - row["bet_worth"]
    else:
        return -row["bet_worth"]


def build_pred_df(
    my_pipeline, X_test, y_test, odds_test, strategy, bankroll=400
):
    test_probs = my_pipeline.predict_proba(X_test)
    preds_test = my_pipeline.predict(X_test)
    labels = my_pipeline.classes_

    print("Classification Report:")
    report = classification_report(y_test, preds_test)
    print(report)

    print("Confusion Matrix:")
    matrix = confusion_matrix(y_test, preds_test, labels=labels)
    print(matrix)

    probs_test_df = pd.DataFrame(
        test_probs,
        index=y_test.index,
        columns=["away_probs", "draw_probs", "home_probs"],
    )

    preds_test_df = pd.DataFrame(preds_test, index=y_test.index, columns=["pred"])

    test_results_df = pd.concat(
        [y_test, preds_test_df, probs_test_df, odds_test], axis=1
    )

    print("\n")
    for l in labels:
        n_times = len(preds_test_df[preds_test_df["pred"] == l])
        print(
            f"Times when {l} was predicted: {n_times} ({round(n_times/len(preds_test_df), 2)})"
        )

    test_results_df["progress"] = bankroll
    test_results_df["current_bankroll"] = bankroll

    for i, row in test_results_df.iterrows():
        odds, probs = get_bet_odds_probs(row)

        previous_bankroll = test_results_df.at[i - 1, "progress"] if i > 0 else bankroll

        bet_worth = get_bet_value(odds, probs, previous_bankroll, strategy=strategy)

        test_results_df.at[i, "bet_worth"] = bet_worth

        profit = get_match_profit(test_results_df.iloc[i])
        
        test_results_df.at[i, "profit"] = profit
        test_results_df.at[i, "progress"] = previous_bankroll + profit

    print("\nTotal bets:", len(test_results_df[test_results_df["profit"] != 0]))
    print("Model profit:", test_results_df.profit.sum())

    negative_consecutive_count = (
        test_results_df["profit"]
        .lt(0)
        .astype(int)
        .groupby((test_results_df["profit"] >= 0).cumsum())
        .sum()
        .max()
    )

    print("Maximum negative sequence: ", negative_consecutive_count)

    positive_consecutive_count = (
        test_results_df["profit"]
        .gt(0)
        .astype(int)
        .groupby((test_results_df["profit"] < 0).cumsum())
        .sum()
        .max()
    )

    print("Maximum positive sequence: ", positive_consecutive_count)
    print("Maximum bet worth:", test_results_df.bet_worth.max())
    print(
        "Minimum bet worth:",
        test_results_df[test_results_df["profit"] != 0].bet_worth.min(),
    )

    return test_results_df


def plot_betting_progress(test_results_df):
    accumulated_values = test_results_df["progress"]

    # Create x-axis values
    x = range(len(accumulated_values))

    # Set the figure size
    plt.figure(figsize=(12, 6))

    # Plot the accumulated column
    plt.plot(x, accumulated_values)

    # Set labels and title
    plt.xlabel("N Bets")
    plt.ylabel("Profit")
    plt.title("Profit by n bets")

    # Display the plot
    plt.show()


def load_saved_utils(dir_path):
    pipeline = load(f"{dir_path}/pipeline.joblib")

    return pipeline


def won_bet(row):
    return 1 if row["pred"] == row["winner"] else 0


def simulate(
    X_train, y_train, X_test, y_test, odds_test, betting_starts_after_n_games, strategy, verbose=1
):
    models_dict = get_models()

    # Only predicting after 15 team games. It's shown that is more profitable
    X_test_filtered = X_test.reset_index(drop=True)[betting_starts_after_n_games:]
    y_test_filtered = y_test.reset_index(drop=True)[betting_starts_after_n_games:]
    odds_test_filtered = odds_test.reset_index(drop=True)[betting_starts_after_n_games:]

    progress_data = []
    # best_results = -9999
    best_results_df = None
    best_pipeline = None

    for model in models_dict.keys():
        print(f"\nResults for model {model}:")

        my_pipeline = build_pipeline(X_train, y_train, models_dict[model]["estimator"])
        if not len(X_test_filtered):
            continue

        test_results_df = build_pred_df(
            my_pipeline,
            X_test_filtered,
            y_test_filtered,
            odds_test_filtered,
            strategy
        )

        if verbose > 1:
            print(test_results_df)
        # if verbose > 1: plot_betting_progress(test_results_df)

        test_results_df["won"] = test_results_df.apply(lambda x: won_bet(x), axis=1)

        total_won = test_results_df[test_results_df["profit"] != 0]["won"].sum()

        progress_data.append(
            [
                test_results_df["profit"].sum(),
                total_won / len(test_results_df[test_results_df["profit"] != 0]),
            ]
        )

        # Define selected model for production
        if model == "knn":
            # best_results = test_results_df["profit"].sum()
            best_results_df = test_results_df
            best_pipeline = my_pipeline

    cols = ["profit", "test_score"]
    profit_df = pd.DataFrame(progress_data, columns=cols, index=models_dict.keys())
    
    if verbose > 0:
        print(profit_df)
    if verbose > 1:
        print(best_results_df.describe())

    # for i, row in best_results_df.iterrows():
    #     print(f"\n{row['home_team']} x {row['away_team']}: {row['pred']}/{row['winner']} {'WON' if row['won'] else ''}")
    #     print(f"Bankroll: {row['progress']}")
    #     print(f"Bet worth: {row['bet_worth']}")
    #     print(f"Profit: {row['profit']}")
    #     print(f"H{row['home_odds']} A{row['away_odds']} D{row['draw_odds']}")

    return best_pipeline