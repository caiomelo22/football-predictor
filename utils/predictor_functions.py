import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_classif
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from joblib import load
from IPython.display import display
import warnings
warnings.filterwarnings('ignore')
print("Setup Complete")

def separate_dataset_info(X):
    y = X.outcome

    odds_cols = ['home_odds', 'away_odds', 'draw_odds']
    odds = X[odds_cols]
    
    for c in odds_cols:
        odds[c] = pd.to_numeric(odds[c], errors='coerce')
    odds.dropna(inplace=True)

    X.drop(['outcome', 'home_score', 'away_score',
                     'home_odds', 'away_odds', 'draw_odds'], axis=1, inplace=True)
    
    _, numerical_cols = set_numerical_categorical_cols(X)
    return X[numerical_cols], y, odds


def get_league_data(league, seasons, season_test):
    # Read the data
    X_full = pd.read_csv(f'./leagues/{league}/formatted_data/{seasons}.csv', index_col=0)
    X_test_full = X_full[X_full['season'] == season_test]
    X_full = X_full[X_full['season'] < season_test]

    # Remove rows with missing target, separate target from predictors
    y = X_full.outcome
    X_full.drop(['outcome', 'home_score', 'away_score', 'home_odds',
                'away_odds', 'draw_odds'], axis=1, inplace=True)

    X_test_full, y_test, odds_test = separate_dataset_info(X_test_full)

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
    X_train = X_full[numerical_cols]
    X_test = X_test_full[numerical_cols]

    return X_train, y_train, X_test


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

def scale_dataset(df, scaler, just_transform = False):
    cols = df.columns
    if just_transform:
        df = pd.DataFrame(scaler.transform(df), columns=cols)
    else:
        df = pd.DataFrame(scaler.fit_transform(df), columns=cols)
    return df

def scale_test_values(X_test, scaler, features_to_explore):
    X_test_scaled = scale_dataset(X_test.loc[:, features_to_explore], scaler, just_transform=True)
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

def train_kmeans(X_train, features_to_explore):
    features_kmeans_list = []
    kmeans_scaler_list = []
    for i in range(len(features_to_explore)):
        kmeans_scaler = MinMaxScaler()
        kmeans = KMeans(n_clusters=5, n_init=10, random_state=0)
        X_train[f"Cluster_{i+1}"] = kmeans.fit_predict(scale_dataset(X_train.loc[:, features_to_explore[i:i+2]], kmeans_scaler))

        X_train[f"Cluster_{i+1}"] = X_train[f"Cluster_{i+1}"].astype('int')

        kmeans_scaler_list.append(kmeans_scaler)
        features_kmeans_list.append((features_to_explore[i:i+2], kmeans))

    return X_train, kmeans_scaler_list, features_kmeans_list

def apply_kmeans(X_test, kmeans_scaler_list, features_kmeans_list):
    for i, fk in enumerate(features_kmeans_list):
        features, kmeans = fk
        X_test[f"Cluster_{i+1}"] = kmeans.predict(scale_dataset(X_test.loc[:, features], kmeans_scaler_list[i], just_transform=True))
        X_test[f"Cluster_{i+1}"] = X_test[f"Cluster_{i+1}"].astype('int')

    return X_test

def create_cluster_features(X_train, X_test, mi_scores):  # First mi scores
    features_to_explore = [mi_scores.index[f] for f in range(len(mi_scores))]
    print('Total features to consider when clustering:', len(features_to_explore))

    X_train, kmeans_scaler_list, features_kmeans_list = train_kmeans(X_train, features_to_explore)
    X_test = apply_kmeans(X_test, kmeans_scaler_list, features_kmeans_list)

    return X_train, X_test, kmeans_scaler_list, features_kmeans_list

def train_pca(X_train, mi_scores, min_mi_score=0.001):
    features_to_explore = [mi_scores.index[f] for f in range(
        len(mi_scores)) if mi_scores[f] > min_mi_score]
    print('Total features to consider when doing the PCA:',
          len(features_to_explore))
    pca = PCA(random_state=0)
    pca_scaler = MinMaxScaler()
    X_pca = pca.fit_transform(scale_dataset(X_train.loc[:, features_to_explore], pca_scaler))

    # Convert to df
    component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
    X_pca = pd.DataFrame(X_pca, columns=component_names, index=X_train.index)

    X_final = pd.concat([X_train, X_pca], axis=1)
    return X_final, features_to_explore, pca_scaler, pca

def apply_pca(X_test, scaler, pca, features_to_explore):
    X_pca = pca.transform(scale_dataset(X_test.loc[:, features_to_explore], scaler, just_transform=True))

    # Convert to df
    component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
    X_pca = pd.DataFrame(X_pca, columns=component_names, index=X_test.index)

    X_final = pd.concat([X_test, X_pca], axis=1)
    return X_final

def apply_pca_datasets(X_train, X_test, mi_scores, min_mi_score=0.001):  # Second mi scores
    X_train, features_to_explore, pca_scaler, pca = train_pca(X_train, mi_scores, min_mi_score=min_mi_score)
    X_test = apply_pca(X_test, pca_scaler, pca, features_to_explore)

    return X_train, X_test, features_to_explore, pca_scaler, pca


def get_models():
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
    
    voting_classifier_estimators = []
    for model in models_dict.keys():
        voting_classifier_estimators.append((model, models_dict[model]['estimator']))
    models_dict['voting_classifier'] = {'estimator': VotingClassifier(estimators=voting_classifier_estimators, voting='soft')}
        
    return models_dict

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
    pipeline = Pipeline(steps=
                           [('preprocessor', preprocessor),
                            # ('feature_selection', SelectKBest(score_func=mutual_info_classif, k='all')),  # Select features based on mutual information
                            # ('pca', PCA(random_state=0), ),  # Perform PCA
                            # ('kmeans', KMeans(n_clusters=5, n_init=10, random_state=0)),  # Perform clustering
                            ('model', model)
                            ])

    # Preprocessing of training data, fit model 
    pipeline.fit(X_train, y_train)
    
    return pipeline

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
    return odds > 1.5
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

def load_saved_utils(league):
    dir_path = f"leagues/{league}/official"
    features_kmeans_list = load(f"{dir_path}/features_kmeans_list.joblib")
    kmeans_scaler_list = load(f"{dir_path}/kmeans_scaler_list.joblib")
    pca_features = load(f"{dir_path}/pca_features.joblib")
    pca_scaler = load(f"{dir_path}/pca_scaler.joblib")
    pca = load(f"{dir_path}/pca.joblib")
    pipeline = load(f"{dir_path}/pipeline.joblib")

    return features_kmeans_list, kmeans_scaler_list, pca_features, pca_scaler, pca, pipeline


def won_bet(row):
    return 1 if row['profit'] > 0 else 0    

def simulate(X_train, y_train, X_test, y_test, odds_test, betting_starts_after_n_games, verbose=1):
    models_dict = get_models()

    # Only predicting after 15 team games. It's shown that is more profitable
    X_test_filtered = X_test.reset_index(drop=True)[betting_starts_after_n_games:]
    y_test_filtered = y_test.reset_index(drop=True)[betting_starts_after_n_games:]
    odds_test_filtered = odds_test.reset_index(drop=True)[betting_starts_after_n_games:]

    progress_data = []
    for model in models_dict.keys():
        print(f"Results for model {model}:")
        my_pipeline = build_pipeline(X_train, y_train, models_dict[model]['estimator'])
        if not len(X_test_filtered): continue
        test_results_df = build_pred_df(my_pipeline, X_test_filtered, y_test_filtered, odds_test_filtered)
        if verbose > 1: display(test_results_df)
        if verbose > 1: plot_betting_progress(test_results_df)
        test_results_df['won'] = test_results_df.apply(lambda x: won_bet(x), axis=1)
        total_won = test_results_df['won'].sum()
        progress_data.append([test_results_df['profit'].sum(), total_won/len(test_results_df)])
        
    cols = ['profit', 'test_score']
    profit_df = pd.DataFrame(progress_data, columns=cols, index=models_dict.keys())
    if verbose > 0: display(profit_df)
    if verbose > 1: display(test_results_df.describe())

    return my_pipeline
