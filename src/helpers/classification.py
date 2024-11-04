import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import load
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
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

    X_filtered = X.drop(
        ["result", "home_score", "away_score", "home_odds", "away_odds", "draw_odds"],
        axis=1,
    )

    _, numerical_cols, _ = set_numerical_categorical_cols(X_filtered)
    return X_filtered[numerical_cols], y, odds


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
        ["result", "home_score", "away_score", "home_odds", "away_odds", "draw_odds"],
        axis=1,
        inplace=True,
    )

    print(X_test_full[["date", "season", "home_team", "away_team", "result"]])
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

def build_pipeline(X_train, y_train, model, preprocess):
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

    steps = []

    if preprocess:
        steps.append(("preprocessor", preprocessor))

    steps.append(("model", model))

    # Bundle preprocessing and modeling code in a pipeline
    pipeline = Pipeline(
        steps=steps
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
    if row["result"] == row["pred"]:
        return (odds * row["bet_worth"]) - row["bet_worth"]
    else:
        return -row["bet_worth"]


def build_pred_df(my_pipeline, X_test, y_test, strategy, bankroll=400):
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
    return 1 if row["pred"] == row["result"] else 0


def get_models(random_state, voting_classifier_models=["logistic_regression"]) -> dict:
    models_dict = {
        "naive_bayes": {
            "estimator": GaussianNB(),
            "params": None,
            "score": None,
        },
        "knn": {
            "estimator": KNeighborsClassifier(n_neighbors=40),
            "params": None,
            "score": None,
        },
        "logistic_regression": {
            "estimator": LogisticRegression(random_state=random_state),
            "params": None,
            "score": None,
        },
        "svm": {
            "estimator": SVC(probability=True, random_state=random_state),
            "params": None,
            "score": None,
        },
        "random_forest_default": {
            "estimator": RandomForestClassifier(random_state=random_state),
            "params": None,
            "score": None,
        },
        "random_forest": {
            "estimator": RandomForestClassifier(
                random_state=random_state, n_estimators=750
            ),
            "params": None,
            "score": None,
        },
        "gradient_boosting": {
            "estimator": GradientBoostingClassifier(random_state=random_state),
            "params": None,
            "score": None,
        },
        "ada_boost": {
            "estimator": AdaBoostClassifier(random_state=random_state),
            "params": None,
            "score": None,
        },
        "mlp": {
            "estimator": MLPClassifier(random_state=0),
            "params": None,
            "score": None,
        },
    }

    voting_classifier_estimators = []

    for model in models_dict.keys():
        if model in voting_classifier_models:
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


def simulate(
    matches,
    start_season,
    season,
    features,
    betting_starts_after_n_games,
    strategy,
    verbose=1,
    random_state=0,
    preprocess=True,
    voting_classifier_models=["logistic_regression"]
):
    matches_filtered = matches[
        (matches["season"] >= start_season) & (matches["season"] <= season)
    ]
    matches_filtered.dropna(subset=features, inplace=True)

    train_set = matches_filtered[matches_filtered["season"] < season]
    test_set = matches_filtered[matches_filtered["season"] == season]

    # Prepare features and labels
    X_train = train_set[features]
    y_train = train_set["result"]

    X_test = test_set[features]
    _ = test_set["result"]

    models_dict = get_models(random_state,voting_classifier_models)

    if not len(X_train):
        return matches, models_dict

    for model in models_dict.keys():
        my_pipeline = build_pipeline(X_train, y_train, models_dict[model]["estimator"], preprocess)
        if not len(X_test):
            continue

        # Predict on the test set
        y_pred = my_pipeline.predict(X_test)
        y_pred_proba = my_pipeline.predict_proba(X_test)  # Get all probabilities

        # Get the order of classes (e.g., ['H', 'D', 'A'])
        class_order = my_pipeline.classes_

        # Map probabilities to correct outcomes based on class order
        home_win_idx = np.where(class_order == "H")[0][0]
        draw_idx = np.where(class_order == "D")[0][0]
        away_win_idx = np.where(class_order == "A")[0][0]

        # Save predictions and probabilities
        matches.loc[X_test.index, f"PredictedRes_{model}"] = y_pred
        matches.loc[X_test.index, f"Proba_HomeWin_{model}"] = y_pred_proba[
            :, home_win_idx
        ]
        matches.loc[X_test.index, f"Proba_Draw_{model}"] = y_pred_proba[:, draw_idx]
        matches.loc[X_test.index, f"Proba_AwayWin_{model}"] = y_pred_proba[
            :, away_win_idx
        ]

        models_dict[model]["pipeline"] = my_pipeline

    return matches, models_dict

# Function to calculate profit based on prediction
def elo_bet_profit(row, start_season, min_odds):
    if row["season"] == start_season:
        return 0
    
    bet_on = None

    if row['home_elo'] > row['away_elo'] and row['home_odds'] > min_odds:
        bet_on = 'H'
        odds = row['home_odds']
    elif row['away_odds'] > min_odds:
        bet_on = 'A'
        odds = row['away_odds']
    
    if bet_on == None:
        return 0
    elif row["result"] == bet_on:
        profit_elo = odds - 1  # Profit from winning bet
    else:
        profit_elo = -1  # Loss from losing bet

    return profit_elo

def home_bet_profit(row, start_season, min_odds):
    if row["season"] == start_season:
        return 0
    
    if row['home_odds'] < min_odds:
        return 0
    elif row["result"] == "H":
        return row['home_odds'] - 1  # Profit from winning bet
    else:
        return -1

def get_bet_value(row, model):
    return 1
    if row[f"PredictedRes_{model}"] == 'H':
        return row[f"Proba_HomeWin_{model}"]
    elif row[f"PredictedRes_{model}"] == 'D':
        return row[f"Proba_Draw_{model}"]
    elif row[f"PredictedRes_{model}"] == 'A':
        return row[f"Proba_AwayWin_{model}"]

def bet_profit_ml(row, model, min_odds):
    if (
        row[f"PredictedRes_{model}"] == None # No prediction
        or pd.isna(row[f"PredictedRes_{model}"]) # No prediction
        or row[f"PredictedRes_{model}"] == 'D' # Exclude draw prediction
        or (row[f"PredictedRes_{model}"] == 'H' and (
            row['home_odds'] < min_odds
            or row['home_odds'] < row[f"Proba_HomeWin_{model}"]
        ))
        or (row[f"PredictedRes_{model}"] == 'D' and (
            row['draw_odds'] < min_odds
            or row['draw_odds'] < row[f"Proba_Draw_{model}"]
        ))
        or (row[f"PredictedRes_{model}"] == 'A' and (
            row['away_odds'] < min_odds
            or row['away_odds'] < row[f"Proba_AwayWin_{model}"]
        ))
    ):
        return 0
    
    bet_value = get_bet_value(row, model)

    if row["result"] == row[f"PredictedRes_{model}"]:
        if row[f"PredictedRes_{model}"] == 'H':
            profit_ml = bet_value*row['home_odds'] - bet_value
        elif row[f"PredictedRes_{model}"] == 'D':
            profit_ml = bet_value*row['draw_odds'] - bet_value
        elif row[f"PredictedRes_{model}"] == 'A':
            profit_ml = bet_value*row['away_odds'] - bet_value
    else:
        profit_ml = -bet_value

    return profit_ml

def get_simulation_results(matches, start_season, min_odds, plot_threshold, random_state):
    # Calculate profits for each model
    matches[f'ProfitElo'] = matches.apply(lambda row: elo_bet_profit(row, start_season, min_odds), axis=1)
    matches[f'CumulativeProfitElo'] = matches[f'ProfitElo'].cumsum()

    matches['ProfitHome'] = matches.apply(lambda row: home_bet_profit(row, start_season, min_odds), axis=1)
    matches['CumulativeProfitHome'] = matches['ProfitHome'].cumsum()

    # Plot cumulative profit
    plt.figure(figsize=(12, 8))

    if matches[f'CumulativeProfitElo'].iloc[-1] > plot_threshold:
        plt.plot(matches["date"], matches[f'CumulativeProfitElo'], label=f'Cumulative Profit Elo')

    if matches[f'CumulativeProfitHome'].iloc[-1] > plot_threshold:
        plt.plot(matches["date"], matches[f'CumulativeProfitHome'], label=f'Cumulative Profit Home')

    model_names = get_models(random_state).keys()

    for model_name in model_names:
        matches[f'ProfitML_{model_name}'] = matches.apply(lambda row: bet_profit_ml(row, model_name, min_odds), axis=1)
        matches[f'CumulativeProfitML_{model_name}'] = matches[f'ProfitML_{model_name}'].cumsum()

        if matches[f'CumulativeProfitML_{model_name}'].iloc[-1] > plot_threshold:
            plt.plot(matches["date"], matches[f'CumulativeProfitML_{model_name}'], label=f'Cumulative Profit ML {model_name}')

    plt.title(f'Cumulative Profit from Betting over {min_odds}')
    plt.xlabel('Game')
    plt.ylabel('Cumulative Profit')
    plt.legend()
    plt.grid(True)
    plt.show()

    # display(matches)

    cum_profit_home = round(matches.iloc[-1]['CumulativeProfitHome'], 4)
    len_home = len(matches[matches['ProfitHome'] != 0])

    print(f"Home method ({cum_profit_home}/{len_home}):", round(cum_profit_home / len_home, 4))

    best_model_name = None
    best_model_profit = -1000

    for model_name in model_names:
        cum_profit_ml = round(matches.iloc[-1][f'CumulativeProfitML_{model_name}'], 4)
        len_ml = len(matches[matches[f"ProfitML_{model_name}"] != 0])

        if cum_profit_ml > best_model_profit:
            best_model_name = model_name
            best_model_profit = cum_profit_ml

        print(f"ML method with {model_name.ljust(20)} --> ({str(cum_profit_ml).rjust(7)}/{str(len_ml).rjust(3)}):", 
        round(cum_profit_ml / len_ml, 4))

    # Evaluate best model
    best_models_predicted_matches = matches[matches[f"PredictedRes_{best_model_name}"].notna()]
    y_pred = best_models_predicted_matches[f"PredictedRes_{best_model_name}"]
    y_test = best_models_predicted_matches["result"]

    print(f"\nProfit for {best_model_name}: ${round(matches.iloc[-1][f'CumulativeProfitML_{best_model_name}'], 4)}")
    print(f"Accuracy for {best_model_name}: {accuracy_score(y_test, y_pred):.2f}")
    print(f"Classification Report for {best_model_name}:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=['H', 'D', 'A'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=['H', 'D', 'A'], yticklabels=['H', 'D', 'A'])
    plt.title(f"Confusion Matrix for {best_model_name}")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    return best_model_name