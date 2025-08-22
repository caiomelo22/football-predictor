import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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


def build_pipeline(X_train, model, preprocess):
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

    return pipeline

def get_bet_unit_value(odds, probs, bankroll, strategy, default_value=1, default_bankroll_pct=0.05):
    if "kelly" in strategy:
        q = 1 - probs  # Probability of losing
        b = odds - 1  # Net odds received on the bet (including the stake)

        if strategy == "half_kelly":
            kelly_fraction = ((probs * b - q) / b) * 0.5
        elif strategy == "quarter_kelly":
            kelly_fraction = ((probs * b - q) / b) * 0.25
        else:
            kelly_fraction = ((probs * b - q) / b)

        return round(
            min(kelly_fraction, 1.0), 4
        )  * bankroll # Limit the bet to 100% of the bankroll
    elif strategy == "bankroll_pct":
        return default_bankroll_pct * bankroll
    else:
        return default_value

def get_bet_value_by_row(row, bankroll, strategy="kelly"):
    odds, probs = get_bet_odds_probs(row)
    return get_bet_unit_value(odds, probs, bankroll, strategy)

def get_bet_odds_probs(bet):
    if bet["pred"] == "H":
        return bet["home_odds"], bet["home_probs"]
    if bet["pred"] == "A":
        return bet["away_odds"], bet["away_probs"]
    if bet["pred"] == "D":
        return bet["draw_odds"], bet["draw_probs"]


def classification_bet_worth_it(prediction, odds, pred_odds, min_odds, bet_value):
    if (
        bet_value < 0 # Value not worth it
        or prediction == None # No prediction
        or pd.isna(prediction) # No prediction
        or odds < min_odds
        # or odds < pred_odds
    ):
        return False

    return True

def get_classification_models(random_state, voting_classifier_models=["logistic_regression"]) -> dict:
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

def simulate_with_classification(
    matches,
    start_season,
    season,
    features,
    random_state=0,
    preprocess=True,
    voting_classifier_models=["logistic_regression"],
    result_col="result",
    class_order=["H", "D", "A"],
):
    matches_filtered = matches[
        (matches["season"] >= start_season) & (matches["season"] <= season)
    ]
    matches_filtered.dropna(subset=features, inplace=True)

    train_set = matches_filtered[matches_filtered["season"] < season]
    test_set = matches_filtered[matches_filtered["season"] == season]

    # Prepare features and labels
    X_train = train_set[features]
    y_train = train_set[result_col]

    X_test = test_set[features]
    y_test = test_set[result_col]

    # Convert y_train e y_test into Categorical before fitting the model
    y_train = pd.Series(y_train, dtype=pd.CategoricalDtype(categories=class_order, ordered=True))
    y_test  = pd.Series(y_test, dtype=pd.CategoricalDtype(categories=class_order, ordered=True))

    models_dict = get_classification_models(random_state,voting_classifier_models)

    new_matches_df = pd.DataFrame(index=X_test.index)

    if not len(X_train):
        return new_matches_df, models_dict

    for model in models_dict.keys():
        my_pipeline = build_pipeline(X_train, models_dict[model]["estimator"], preprocess)
        
        my_pipeline.fit(X_train, y_train)  # Fit the model on the training set
        
        if not len(X_test):
            continue

        # Predict on the test set
        y_pred = my_pipeline.predict(X_test)
        y_pred_proba = my_pipeline.predict_proba(X_test)  # Get all probabilities

        new_matches_df.loc[X_test.index, f"pred_{result_col}_{model}"] = y_pred

        for class_str in class_order:
            y_pred_index = class_order.index(class_str)

            # Save the predicted probabilities for each class
            new_matches_df.loc[X_test.index, f"proba_{result_col}_{class_str}_{model}"] = y_pred_proba[:, y_pred_index]

        models_dict[model]["pipeline"] = my_pipeline

    return new_matches_df, models_dict

# Function to calculate profit based on prediction
def elo_bet_profit(row, start_season, min_odds, bankroll, strategy, default_value, default_bankroll_pct, odds_col_suffix="odds", result_col="result"):
    if row["season"] == start_season:
        return 0
    
    bet_on = None

    if row['home_elo'] > row['away_elo'] and row[f'home_{odds_col_suffix}'] > min_odds:
        bet_on = 'H'
        odds = row[f'home_{odds_col_suffix}']
    elif row[f'away_{odds_col_suffix}'] > min_odds:
        bet_on = 'A'
        odds = row[f'away_{odds_col_suffix}']
    
    if bet_on == None or row[result_col] == "P":
        return 0

    bet_value = get_bet_unit_value(odds, 1, bankroll, strategy, default_value, default_bankroll_pct)
    
    if row[result_col] == bet_on:
        profit_elo = (odds*bet_value) - bet_value  # Profit from winning bet
    else:
        profit_elo = - bet_value  # Loss from losing bet

    return profit_elo

def home_bet_profit(row, start_season, min_odds, bankroll, strategy, default_value, default_bankroll_pct, odds_col_suffix="odds", result_col="result"):
    if row["season"] == start_season:
        return 0

    bet_value = get_bet_unit_value(row['home_odds'], 1, bankroll, strategy, default_value, default_bankroll_pct)
    
    if row[f'home_{odds_col_suffix}'] < min_odds or row[result_col] == "P":
        return 0
    elif row[result_col] == "H":
        return (row[f'home_{odds_col_suffix}'] * bet_value) - bet_value  # Profit from winning bet
    else:
        return - bet_value

def get_pred_text(pred_str):
    if pred_str == "H":
        return "home"
    elif pred_str == "A":
        return "away"
    elif pred_str == "D":
        return "draw"
    elif pred_str == "O":
        return "overs"
    elif pred_str == "U":
        return "unders"
    
def get_betting_line(pred, result_col="result"):
    if result_col == "ahc_result":
        if pred == "H":
            return "home_ahc_odds"
        elif pred == "A":
            return "away_ahc_odds"
    elif result_col == "totals_result":
        if pred == "O":
            return "overs_odds"
        elif pred == "U":
            return "unders_odds"
        
    return None

def get_adjusted_profit(adjusted, bet_value, selected_odds):
    if adjusted > 0.5:  # Full win
        return bet_value * (selected_odds - 1)
    elif 0 < adjusted <= 0.5:  # Half win
        return (bet_value * (selected_odds - 1)) / 2
    elif adjusted == 0:  # Push (stake refunded)
        return 0
    elif -0.5 <= adjusted < 0:  # Half loss
        return -bet_value / 2
    else:  # Full loss
        return -bet_value

def get_ahc_profit(row, pred, bet_value, selected_odds):
    # Handicap line always refers to the Away team
    line = row["ahc_line"]
    actual_diff = row["home_score"] - row["away_score"]

    # If bet is on Away, use the line as it is
    if pred == "A":
        adjusted = -actual_diff + line
    else:  # If bet is on Home, invert the line perspective
        adjusted = actual_diff - line

    # Now evaluate outcome
    return get_adjusted_profit(adjusted, bet_value, selected_odds)


def get_totals_profit(row, pred, bet_value, selected_odds):
    line = row["totals_line"]
    actual_total = row["home_score"] + row["away_score"]

    # If Over, adjust perspective
    if pred == "O":
        adjusted = actual_total - line
    else:  # Under
        adjusted = line - actual_total

    # Now evaluate outcome
    return get_adjusted_profit(adjusted, bet_value, selected_odds)

def get_profit_classification(row, model, min_odds, bankroll, strategy="kelly", default_value=1, default_bankroll_pct=0.05, odds_col_suffix="odds", result_col="result"):
    if "P" in [row[result_col], row[f"pred_{result_col}_{model}"]]:
        return 0    
    
    pred = row[f"pred_{result_col}_{model}"]

    if pred == None or pd.isna(pred):
        return 0

    pred_text = get_pred_text(pred)
    
    selected_odds = row[f'{pred_text}_{odds_col_suffix}']
    selected_pred_odds = row[f"proba_{result_col}_{pred}_{model}"]

    bet_value = get_bet_unit_value(selected_odds, selected_pred_odds, bankroll, strategy, default_value, default_bankroll_pct)

    if not classification_bet_worth_it(
        row[f"pred_{result_col}_{model}"],
        selected_odds,
        selected_pred_odds,
        min_odds,
        bet_value,
    ):
        return 0

    # Handle standard 1X2 market
    if result_col == "result":
        if row[result_col] == row[f"pred_{result_col}_{model}"]:
            return bet_value * selected_odds - bet_value
        return -bet_value

    # Handle AHC market
    if result_col == "ahc_result":
        return get_ahc_profit(row, pred, bet_value, selected_odds)
    
    # Handle Totals market
    if result_col == "totals_result":
        return get_totals_profit(row, pred, bet_value, selected_odds)

    return 0  # Default return if no conditions met

def calculate_baseline_classification_results(matches, start_season, min_odds, plot_threshold, bankroll, strategy, default_value, default_bankroll_pct, odds_col_suffix="odds", result_col="result"):
    # Calculate profits for each model
    matches[f'profit_elo_{result_col}'] = matches.apply(lambda row: elo_bet_profit(row, start_season, min_odds, bankroll, strategy, default_value, default_bankroll_pct, odds_col_suffix, result_col), axis=1)
    matches[f'cum_profit_elo_{result_col}'] = matches[f'profit_elo_{result_col}'].cumsum()

    matches[f'profit_home_{result_col}'] = matches.apply(lambda row: home_bet_profit(row, start_season, min_odds, bankroll, strategy, default_value, default_bankroll_pct, odds_col_suffix, result_col), axis=1)
    matches[f'cum_profit_home_{result_col}'] = matches[f'profit_home_{result_col}'].cumsum()

    # Plot cumulative profit
    plt.figure(figsize=(12, 8))

    if matches[f'cum_profit_elo_{result_col}'].iloc[-1] > plot_threshold:
        plt.plot(matches["date"], matches[f'cum_profit_elo_{result_col}'], label=f'Cumulative Profit Elo {result_col.capitalize()}')

    if matches[f'cum_profit_home_{result_col}'].iloc[-1] > plot_threshold:
        plt.plot(matches["date"], matches[f'cum_profit_home_{result_col}'], label=f'Cumulative Profit Home {result_col.capitalize()}')

def display_baseline_classification_results(matches, result_col="result"):
    result_col_capitalized = result_col.capitalize()

    def display_baseline_result(matches, result_col, method):
        cum_profit = round(matches.iloc[-1][f'cum_profit_{method}_{result_col}'], 4)
        len_method = len(matches[matches[f'profit_{method}_{result_col}'] != 0])
        avg_profit = round(cum_profit / len_method, 4) if len_method > 0 else 0

        print(f"{result_col_capitalized} {method.capitalize()} method ({cum_profit}/{len_method}):", avg_profit)

    # Display cumulative profit for Elo method
    display_baseline_result(matches, result_col, "elo")
    
    # Display cumulative profit for Home method
    display_baseline_result(matches, result_col, "home")

def display_market_classification_results(matches, start_season, min_odds, plot_threshold, bankroll, strategy, default_value, default_bankroll_pct, odds_col_suffix="odds", result_col = "result", class_order=["H", "D", "A"], include_baseline=True):
    result_col_capitalized = result_col.capitalize()

    # Get baseline classification results
    if include_baseline:
        calculate_baseline_classification_results(matches, start_season, min_odds, plot_threshold, bankroll, strategy, default_value, default_bankroll_pct, odds_col_suffix=odds_col_suffix, result_col=result_col)   
    
    model_names = get_classification_models(random_state=0).keys()

    for model_name in model_names:
        matches[f'profit_{result_col}_{model_name}'] = matches.apply(lambda row: get_profit_classification(row, model_name, min_odds, bankroll, strategy, default_value, default_bankroll_pct, odds_col_suffix, result_col), axis=1)
        matches[f'cum_profit_{result_col}_{model_name}'] = matches[f'profit_{result_col}_{model_name}'].cumsum()

        if matches[f'cum_profit_{result_col}_{model_name}'].iloc[-1] > plot_threshold:
            plt.plot(matches["date"], matches[f'cum_profit_{result_col}_{model_name}'], label=f'Cumulative Profit {result_col_capitalized} {model_name}')

    plt.title(f'Cumulative {result_col_capitalized} Profit from Betting over {min_odds}')
    plt.xlabel('Game')
    plt.ylabel(f'Cumulative {result_col_capitalized} Profit')
    plt.legend()
    plt.grid(True)
    plt.show()

    if include_baseline:
        display_baseline_classification_results(matches, result_col=result_col)

    best_model_name = None
    best_model_profit = -1000

    for model_name in model_names:
        cum_profit = round(matches.iloc[-1][f'cum_profit_{result_col}_{model_name}'], 4)
        col_size = len(matches[matches[f"profit_{result_col}_{model_name}"] != 0])

        if cum_profit > best_model_profit:
            best_model_name = model_name
            best_model_profit = cum_profit

        print(f"{result_col_capitalized} method with {model_name.ljust(20)} --> ({str(cum_profit).rjust(7)}/{str(col_size).rjust(3)}):", 
        round(cum_profit / col_size, 4))
        
    # Evaluate best model
    best_models_predicted_matches = matches[matches[f"pred_{result_col}_{best_model_name}"].notna()]
    y_pred = best_models_predicted_matches[f"pred_{result_col}_{best_model_name}"]
    y_test = best_models_predicted_matches[result_col]

    print(f"\n{result_col_capitalized} Profit for {best_model_name}: ${round(matches.iloc[-1][f'cum_profit_{result_col}_{best_model_name}'], 4)}")
    print(f"{result_col_capitalized} Accuracy for {best_model_name}: {accuracy_score(y_test, y_pred):.2f}")
    print(f"{result_col_capitalized} Classification Report for {best_model_name}:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=class_order)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_order, yticklabels=class_order)
    plt.title(f"Confusion Matrix for {result_col_capitalized} - {best_model_name}")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    return best_model_name

def get_classification_simulation_results(matches, start_season, plot_threshold, bankroll, strategy, default_value, default_bankroll_pct, min_odds_1x2=2.2, min_odds_ahc=1.9, min_odds_totals=1.7):
    # Display 1x2 classification results
    print("-"* 50)
    print("1x2 Classification Results")
    best_1x2_model = display_market_classification_results(
        matches,
        start_season,
        min_odds_1x2,
        plot_threshold,
        bankroll,
        strategy,
        default_value,
        default_bankroll_pct,
        odds_col_suffix="odds",
        result_col="result",
        class_order=["H", "A", "D"],
        include_baseline=True,
    )

    # Display AHC classification results
    print("-"* 50)
    print("AHC Classification Results")
    best_ahc_model = display_market_classification_results(
        matches,
        start_season,
        min_odds_ahc,
        plot_threshold,
        bankroll,
        strategy,
        default_value,
        default_bankroll_pct,
        odds_col_suffix="ahc_odds",
        result_col="ahc_result",
        class_order=["H", "A", "P"],
        include_baseline=True,
    )

    # Display Totals classification results
    print("-"* 50)
    print("Totals Classification Results")
    best_totals_model = display_market_classification_results(
        matches,
        start_season,
        min_odds_totals,
        plot_threshold,
        bankroll,
        strategy,
        default_value,
        default_bankroll_pct,
        odds_col_suffix="odds",
        result_col="totals_result",
        class_order=["O", "U", "P"],
        include_baseline=False,
    )

    return best_1x2_model, best_ahc_model, best_totals_model

def display_random_forest_feature_importances(last_season_models, features):
    for market, models_dict in last_season_models.items():        
        print(f"Feature Importances for {market.capitalize()}:")

        pipeline = models_dict["random_forest_default"]["pipeline"]
        
        # Get the Random Forest model from the matches DataFrame
        importances = pipeline.named_steps['model'].feature_importances_

        # Create a DataFrame for feature importances
        feature_importances = pd.DataFrame({
            'feature': features,
            'importance': importances
        }).sort_values(by='importance', ascending=False)

        # Plot feature importances
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=feature_importances)
        plt.title(f'Feature Importances for {market.capitalize()}')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.show()