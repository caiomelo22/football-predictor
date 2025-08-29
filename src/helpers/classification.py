import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

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

class LabelEncoderClassifier(BaseEstimator, ClassifierMixin):
    """
    Wrapper that encodes string labels to integer codes for estimators that
    require numeric labels (e.g. XGBClassifier), and decodes predictions back
    to the original labels. Exposes estimator's params as estimator__* so it's
    compatible with RandomizedSearchCV.
    """
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y, **fit_params):
        # ensure y is string-like before encoding
        y_ser = pd.Series(y).astype(str)
        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(y_ser)
        self.estimator.fit(X, y_enc, **fit_params)
        self.classes_ = self.le_.classes_
        return self

    def predict(self, X):
        preds = self.estimator.predict(X)
        return self.le_.inverse_transform(preds.astype(int))

    def predict_proba(self, X):
        # returns probabilities in the same column order as the wrapped estimator
        # wrapped estimator was trained on encoded classes (0..n-1) so columns align
        probs = self.estimator.predict_proba(X)
        return probs

    def get_params(self, deep=True):
        params = {"estimator": self.estimator}
        if deep:
            nested = self.estimator.get_params()
            for k, v in nested.items():
                params[f"estimator__{k}"] = v
        return params

    def set_params(self, **params):
        if "estimator" in params:
            self.estimator = params.pop("estimator")
        # forward estimator__ params
        est_params = {k.split("estimator__")[1]: v for k, v in params.items() if k.startswith("estimator__")}
        if est_params:
            self.estimator.set_params(**est_params)
        return self

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
    
    # wrap XGBClassifier so string labels are handled automatically
    if isinstance(model, XGBClassifier):
        model = LabelEncoderClassifier(model)

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
        or odds < pred_odds
    ):
        return False

    return True

def get_classification_models(random_state=0, voting_models=[], manual_params=None) -> dict:
    models_dict = {
        "naive_bayes": {
            "estimator": GaussianNB(),
            "params": None,
            "score": None,
            "param_grid": {},
        },
        "knn": {
            "estimator": KNeighborsClassifier(),
            "params": None,
            "score": None,
            "param_grid": {
                "model__n_neighbors": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],  # More granular
                "model__weights": ["uniform", "distance"],  # Add weights parameter
                "model__metric": ["euclidean", "manhattan"]  # Add distance metrics
            }
        },
        "logistic_regression": {
            "estimator": LogisticRegression(random_state=random_state, max_iter=200),
            "params": None,
            "score": None,
            "param_grid": {
                "model__C": [0.001, 0.01, 0.1, 1, 5, 10, 50, 100],
            },
        },
        "svm": {
            "estimator": SVC(probability=True, random_state=random_state),
            "params": None,
            "score": None,
            "param_grid": {},
        },
        "random_forest": {
            "estimator": RandomForestClassifier(random_state=random_state, n_jobs=-1),
            "params": None,
            "score": None,
            "param_grid": {
                "model__n_estimators": [100, 200, 300, 500, 1000],  # Reduced range but more focused
                "model__max_depth": [5, 10, 15, 20, None],  # More reasonable depths
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", "log2"],  # Removed 0.6 as it's less common
                "model__class_weight": [None, "balanced"]  # Simplified
            }
        },
        "gradient_boosting": {
            "estimator": GradientBoostingClassifier(random_state=random_state),
            "params": None,
            "score": None,
            "param_grid": {
                "model__n_estimators": [10, 50, 100, 200, 300],
                "model__learning_rate": [0.01, 0.05, 0.1],
                "model__max_depth": [2, 3, 4],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__subsample": [0.7, 0.85, 1.0]
            }
        },
        "ada_boost": {
            "estimator": AdaBoostClassifier(random_state=random_state),
            "params": None,
            "score": None,
            "param_grid": {
                "model__n_estimators": [10, 50, 100, 200, 300],
                "model__learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
                "model__algorithm": ["SAMME.R", "SAMME"]
            }
        },
        "mlp": {
            "estimator": MLPClassifier(random_state=random_state, max_iter=400),
            "params": None,
            "score": None,
            "param_grid": {},
        },
        "xgboost": {
            "estimator": XGBClassifier(random_state=random_state, use_label_encoder=False, eval_metric="mlogloss"),
            "params": None,
            "score": None,
            "param_grid": {
                "model__estimator__n_estimators": [100, 200, 300],  # CORRECT: model__estimator__ prefix
                "model__estimator__learning_rate": [0.01, 0.1, 0.2],
                "model__estimator__max_depth": [3, 6, 9],
                "model__estimator__subsample": [0.8, 1.0],
            },
        },
    }
    
    # Apply manual parameters if provided
    if manual_params:
        for model_name, params in manual_params.items():
            if model_name in models_dict:
                models_dict[model_name]["estimator"].set_params(**params)
                models_dict[model_name]["params"] = params

    if voting_models:
        estimators = [(model, models_dict[model]["estimator"]) for model in voting_models if model in models_dict]
        if estimators:
            voting_clf = VotingClassifier(estimators=estimators, voting='soft')
            models_dict["voting"] = {
                "estimator": voting_clf,
                "params": None,
                "score": None,
                "param_grid": {}
            }

    return models_dict

def get_filtered_matches(
    matches,
    features,
    start_season,
    season
):
    filtered_matches = matches[
        (matches["season"] >= start_season) & (matches["season"] <= season)
    ]
    filtered_matches.dropna(subset=features, inplace=True)

    return filtered_matches

def simulate_with_classification(
    matches,
    start_season,
    season,
    features,
    random_state=0,
    preprocess=True,
    result_col="result",
    class_order=["H", "D", "A"],
    fast_simulation=True,
    custom_models_config=None,
    voting_models=[],
):
    matches_filtered = get_filtered_matches(
        matches,
        features,
        start_season,
        season
    )

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

    manual_params = custom_models_config if fast_simulation else None

    models_dict = get_classification_models(random_state, manual_params=manual_params, voting_models=voting_models)

    new_matches_df = pd.DataFrame(index=X_test.index)

    if not len(X_train):
        return new_matches_df, models_dict

    for model in models_dict.keys():
        my_pipeline = build_pipeline(
            X_train,
            models_dict[model]["estimator"],
            preprocess,
        )

        param_grid = models_dict[model].get("param_grid", {})

        if not fast_simulation and param_grid:            
            total_combinations = 1
            for param_values in param_grid.values():
                total_combinations *= len(param_values)

            cv_splitter = TimeSeriesSplit(n_splits=3)  # Respects chronological order

            if total_combinations <= 50:
                search = GridSearchCV(
                    my_pipeline,
                    param_grid=param_grid,
                    cv=cv_splitter,  # Increased CV folds
                    scoring="accuracy",
                    n_jobs=-1,
                    refit=True
                )
            else:
                search = RandomizedSearchCV(
                    my_pipeline,
                    param_distributions=param_grid,
                    n_iter=50,
                    cv=cv_splitter,
                    scoring="accuracy",
                    n_jobs=-1,
                    random_state=random_state,
                    refit=True
                )

            search.fit(X_train, y_train)
            best_pipeline = search.best_estimator_
            models_dict[model]["params"] = search.best_params_
            my_pipeline = best_pipeline
        else:
            my_pipeline.fit(X_train, y_train)
            models_dict[model]["params"] = None
        
        models_dict[model]["pipeline"] = my_pipeline
        
        if not len(X_test):
            continue

        # Predict on the test set
        y_pred = my_pipeline.predict(X_test)
        y_pred_proba = my_pipeline.predict_proba(X_test)  # Get all probabilities

        new_matches_df.loc[X_test.index, f"pred_{result_col}_{model}"] = y_pred

        for class_str in class_order:
            if class_str not in my_pipeline.classes_:
                continue
            
            class_idx = list(my_pipeline.classes_).index(class_str)

            # Save the predicted probabilities for each class
            new_matches_df.loc[X_test.index, f"proba_{result_col}_{class_str}_{model}"] = y_pred_proba[:, class_idx]

    return new_matches_df, models_dict

def get_classification_accuracy(matches, model, result_col="result"):
    """
    Return accuracy for a single classification model.
    """
    pred_col = f"pred_{result_col}_{model}"
    df = matches.dropna(subset=[result_col, pred_col])
    if df.empty:
        return 0.0
    y_true = df[result_col].astype(str)
    y_pred = df[pred_col].astype(str)
    return accuracy_score(y_true, y_pred)


def show_classification_accuracies(matches, selected_models, result_col="result"):
    """
    Print accuracies for selected models sorted by accuracy desc.
    """
    scores = {}
    for model in selected_models:
        acc = get_classification_accuracy(matches, model, result_col)
        selected_models[model]["score"] = acc
        scores[model] = acc

    for model, acc in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        print(f"{model}: Accuracy = {acc:.4f}")

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

def display_market_classification_results(matches, start_season, min_odds, plot_threshold, bankroll, strategy, default_value, default_bankroll_pct, odds_col_suffix="odds", result_col = "result", class_order=["H", "D", "A"], include_baseline=True, logger=None, voting_models=[]):
    result_col_capitalized = result_col.capitalize()

    # Add this line to set the figure size to 12x8
    plt.figure(figsize=(12, 8))

    # Get baseline classification results
    if include_baseline:
        calculate_baseline_classification_results(matches, start_season, min_odds, plot_threshold, bankroll, strategy, default_value, default_bankroll_pct, odds_col_suffix=odds_col_suffix, result_col=result_col)   
    
    model_names = get_classification_models(random_state=0, voting_models=voting_models).keys()
    
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

    # Save chart if logger is provided
    if logger:
        logger.save_chart(f"profit_analysis_{result_col}", f"Profit Analysis for {result_col.capitalize()}")
    
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
    
    # Save chart if logger is provided
    if logger:
        logger.save_chart(f"confusion_matrix_{result_col}", f"Confusion Matrix for {result_col.capitalize()}")

    plt.show()

    return best_model_name

def get_classification_simulation_results(matches, start_season, plot_threshold, bankroll, strategy, default_value, default_bankroll_pct, min_odds_1x2=2.2, min_odds_ahc=1.9, min_odds_totals=1.7, logger=None, voting_models=[]):
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
        logger=logger,
        voting_models=voting_models
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
        logger=logger,
        voting_models=voting_models
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
        logger=logger,
        voting_models=voting_models
    )

    return best_1x2_model, best_ahc_model, best_totals_model

def display_random_forest_feature_importances(last_season_models, features, logger=None):
    for market, models_dict in last_season_models.items():        
        print(f"Feature Importances for {market.capitalize()}:")

        pipeline = models_dict["random_forest"]["pipeline"]

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

        # Save chart if logger is provided
        if logger:
            logger.save_chart(f"feature_importance_{market}", f"Feature Importance for {market.capitalize()}")
        
        plt.show()