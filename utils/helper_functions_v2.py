import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_classif
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import os
from joblib import dump
import warnings
from scipy.stats import uniform, randint
warnings.filterwarnings('ignore')
print("Setup Complete")


def get_league_data(league, seasons, season_test):
    # Read the data
    X_full = pd.read_csv(f'./leagues_v2/{league}/formatted_data/{seasons}.csv', index_col=0)
    X_test_full = X_full[X_full['season'] == season_test]
    X_full = X_full[X_full['season'] < season_test]

    # Remove rows with missing target, separate target from predictors
    y = X_full.outcome
    X_full.drop(['outcome', 'home_score', 'away_score', 'home_odds',
                'away_odds', 'draw_odds'], axis=1, inplace=True)

    y_test = X_test_full.outcome

    odds_cols = ['home_odds', 'away_odds', 'draw_odds']
    odds_test = X_test_full[odds_cols]
    
    for c in odds_cols:
        odds_test[c] = pd.to_numeric(odds_test[c], errors='coerce')
    odds_test.dropna(inplace=True)

    X_test_full.drop(['outcome', 'home_score', 'away_score',
                     'home_odds', 'away_odds', 'draw_odds'], axis=1, inplace=True)

    return X_full, y, X_test_full, y_test, odds_test


def set_numerical_categorical_cols(X):
    # "Cardinality" means the number of unique values in a column
    # Select categorical columns with relatively low cardinality (convenient but arbitrary)
    categorical_cols = [cname for cname in X.columns if
                        X[cname].nunique() < 10 and
                        X[cname].dtype == "object"]

    # Select numerical columns
    numerical_cols = [cname for cname in X.columns if
                      X[cname].dtype in ['int64', 'float64']]

    return categorical_cols, numerical_cols


def filter_datasets(X_full, y, X_test_full, categorical_cols, numerical_cols):
    # Keep selected columns only
    my_cols = categorical_cols + numerical_cols
    y_train = y.copy()
    X_train = X_full[my_cols]
    X_test = X_test_full[my_cols]

    return X_train, y_train, X_test


def transform_x(X, categorical_cols, numerical_cols):
    X = X.copy()

    # Replacing missing values
    imputer = SimpleImputer(strategy='mean')
    X[numerical_cols] = imputer.fit_transform(X[numerical_cols])

    if categorical_cols:
        imputer = SimpleImputer(strategy='constant')
        X[categorical_cols] = imputer.fit_transform(X[categorical_cols])

    # Label encoding for categoricals
    for colname in categorical_cols:
        X[colname], _ = X[colname].factorize()

    return X


def make_mi_scores(X, y):
    # All discrete features should now have integer dtypes
    discrete_features = X.dtypes == int

    mi_scores = mutual_info_classif(
        X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name='MI Scores', index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


def plot_mi_scores(scores):
    plt.figure(figsize=(12, 12))
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")


def plot_feature_corr_chart(X, numerical_cols):
    plt.figure(figsize=(12, 12))
    plot_data = X[numerical_cols].corr()
    sns.heatmap(data=plot_data)
    plt.show()


def scale_values(X, X_test, features_to_explore):
    # Standardize
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X.loc[:, features_to_explore])
    X_test_scaled = scaler.transform(X_test.loc[:, features_to_explore])
    return X_scaled, X_test_scaled


def create_cluster_features(X_train, X_test, mi_scores):  # First mi scores
    features_to_explore = [mi_scores.index[f] for f in range(len(mi_scores))]
    print('Total features to consider when clustering:', len(features_to_explore))

    for i in range(len(features_to_explore)):
        X_scaled, X_test_scaled = scale_values(X_train, X_test, features_to_explore[i:i+2])

        kmeans = KMeans(n_clusters=5, n_init=10, random_state=0)
        X_train[f"Cluster_{i+1}"] = kmeans.fit_predict(X_scaled)
        X_test[f"Cluster_{i+1}"] = kmeans.predict(X_test_scaled)

        X_train[f"Cluster_{i+1}"] = X_train[f"Cluster_{i+1}"].astype('int')
        X_test[f"Cluster_{i+1}"] = X_test[f"Cluster_{i+1}"].astype('int')


def get_pca(X, X_scaled, pca, just_transform=False):
    if just_transform:
        X_pca = pca.transform(X_scaled)
    else:
        X_pca = pca.fit_transform(X_scaled)

    # Convert to df
    component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
    X_pca = pd.DataFrame(X_pca, columns=component_names, index=X.index)

    X_final = pd.concat([X, X_pca], axis=1)
    return X_final


def apply_pca_datasets(X_train, X_test, mi_scores, min_mi_score=0.001):  # Second mi scores
    features_to_explore = [mi_scores.index[f] for f in range(
        len(mi_scores)) if mi_scores[f] > min_mi_score]
    print('Total features to consider when doing the PCA:',
          len(features_to_explore))
    
    # Standardize
    X_scaled, X_test_scaled = scale_values(X_train, X_test, features_to_explore)

    pca = PCA(random_state=0)
    X_train = get_pca(X_train, X_scaled, pca)
    X_test = get_pca(X_test, X_test_scaled, pca, just_transform=True)

    return X_train, X_test


def run_random_search(X_train, y_train, season, league):
    models = {
        # 'random_forest': RandomForestClassifier(random_state=0),
    }
    
    param_grid = {
        'random_forest': {
            'n_estimators': [100, 200, 500, 1000, 1500, 2000],
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 5, 10, 25, 50],
            'min_samples_split': [2, 5, 10, 25, 50],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        },
    }

    models_dict = {
        'naive_bayes': {
            'estimator': GaussianNB(var_smoothing=1e-7),
            'params': None,
            'score': None
        },
        'knn': {
            'estimator': KNeighborsClassifier(n_neighbors=10),
            'params': None,
            'score': None
        },
        'logistic_regression': {
            'estimator': LogisticRegression(random_state=0),
            'params': None,
            'score': None
        }
    }
    
    for model_name, model in models.items():
        print(f"\nRunning random search for {model_name} in the season {season}")
        
        param_grid_model = param_grid.get(model_name)
        if param_grid_model is None:
            continue
        
        random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid_model, cv=5)
        random_search.fit(X_train, y_train)
        
        best_model = random_search.best_estimator_
        best_params = random_search.best_params_
        best_score = random_search.best_score_
        
        print(f"Best parameters: {best_params}")
        print(f"Best score: {best_score}")

        models_dict[model_name] = {
            'estimator': best_model,
            'params': best_params,
            'score': best_score
        }
        
    save_model_results(models_dict, season, league)

def build_pipeline(X_train, y_train, model):
    # Preprocessing for numerical data
    numerical_transformer = SimpleImputer(strategy='median')

    categorical_cols, numerical_cols = set_numerical_categorical_cols(X_train)

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

    # Define normalization function
    scaler = MinMaxScaler()

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols),
            # ('normalization', scaler, numerical_cols + categorical_cols)
        ])

    # Bundle preprocessing and modeling code in a pipeline
    my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('model', model)
                                ])

    # Preprocessing of training data, fit model 
    my_pipeline.fit(X_train, y_train)
    
    return my_pipeline

def get_pred_odds(probs):
    return 1/probs

def get_bet_value(odds, probs, bankroll):
    return bankroll * 0.05
    return (bankroll*(probs - ((1-probs)/odds)))/6

def get_bet_odds_probs(bet):
    if bet['pred'] == 'H': return bet['home_odds'], bet['home_probs']
    if bet['pred'] == 'A': return bet['away_odds'], bet['away_probs']
    if bet['pred'] == 'D': return bet['draw_odds'], bet['draw_probs']

def bet_worth_it(probs, odds):
    return True
    return get_pred_odds(probs) > odds and odds > 1.5
    
def get_bet_value_by_row(row, bankroll):
    odds, probs = get_bet_odds_probs(row)
    return get_bet_value(odds, probs, bankroll)

def get_match_profit(row, bankroll):
    odds, probs = get_bet_odds_probs(row)
    bet_value = get_bet_value(probs=probs, odds=odds, bankroll=bankroll)
    if row['outcome'] == row['pred']:
        if bet_worth_it(probs, odds): return (odds*bet_value) - bet_value
        else: return 0
    else:
        return -bet_value

def build_pred_df(my_pipeline, X_test, y_test, odds_test, bankroll=2000):
    preds_test = my_pipeline.predict(X_test)

    report = classification_report(y_test, preds_test)
    print('Classification Report:')
    print(report)

    labels = ["H", "D", "A"]
    matrix = confusion_matrix(y_test, preds_test, labels=labels)
    print('Confusion Matrix:')
    print(matrix)

    test_probs = my_pipeline.predict_proba(X_test)
    probs_test_df = pd.DataFrame(test_probs, index=y_test.index, columns=['away_probs', 'draw_probs', 'home_probs'])
    preds_test_df = pd.DataFrame(preds_test, index=y_test.index, columns=['pred'])
    test_results_df = pd.concat([y_test, preds_test_df, probs_test_df, odds_test], axis=1)

    print('\n')
    for l in labels:
        n_times = len(preds_test_df[preds_test_df['pred'] == l])
        print(f"Times when {l} was predicted: {n_times} ({round(n_times/len(preds_test_df), 2)})")

    test_results_df['bet_worth'] = test_results_df.apply(lambda x: get_bet_value_by_row(x, bankroll), axis=1)
    test_results_df['profit'] = test_results_df.apply(lambda x: get_match_profit(x, bankroll), axis=1)
    test_results_df['progress'] = [bankroll] + test_results_df['profit'].cumsum().add(bankroll).tolist()[1:]

    print('\nModel profit:', test_results_df.profit.sum())
    negative_consecutive_count = test_results_df['profit'].lt(0).astype(int).groupby((test_results_df['profit'] >= 0).cumsum()).sum().max()
    print('Maximum negative sequence: ', negative_consecutive_count)
    positive_consecutive_count = test_results_df['profit'].gt(0).astype(int).groupby((test_results_df['profit'] < 0).cumsum()).sum().max()
    print('Maximum positive sequence: ', positive_consecutive_count)
    print('Maximum bet worth:', test_results_df.bet_worth.max())
    print('Minimum bet worth:', test_results_df.bet_worth.min())

    return test_results_df

def save_model_results(models_dict, season, league):
    path = f"leagues_v2/{league}/best_models/{season}"
    if not os.path.exists(path):
        os.makedirs(path)

    for model in models_dict.keys():
        model_path = f"{path}/{model}.joblib"
        dump(models_dict[model]['estimator'], model_path)
        del models_dict[model]['estimator']

    json_path = f"{path}/best_models.json"
    with open(json_path, "w") as file:
        json.dump(models_dict, file)

def plot_betting_progress(test_results_df):
    accumulated_values = test_results_df['progress']

    # Create x-axis values
    x = range(len(accumulated_values))

    # Set the figure size
    plt.figure(figsize=(12, 6))

    # Plot the accumulated column
    plt.plot(x, accumulated_values)

    # Set labels and title
    plt.xlabel('N Bets')
    plt.ylabel('Profit')
    plt.title('Profit by n bets')

    # Display the plot
    plt.show()
