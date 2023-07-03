import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_classif
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from joblib import load
from . import league_options as lo
from IPython.display import display
import warnings
from keras.models import Sequential
from keras.layers import Dense

from sklearn.svm import SVC
warnings.filterwarnings('ignore')
print("Setup Complete")

def separate_dataset_info(X):
    y = X.winner

    odds_cols = ['date','season','home_team','away_team','home_odds', 'away_odds', 'draw_odds']
    odds = X[odds_cols]
    
    for c in odds_cols:
        if 'odds' in c:
            odds[c] = pd.to_numeric(odds[c], errors='coerce')
    odds.dropna(inplace=True)

    X.drop(['winner', 'home_score', 'away_score',
                     'home_odds', 'away_odds', 'draw_odds'], axis=1, inplace=True)
    
    _, numerical_cols, _ = set_numerical_categorical_cols(X)
    return X[numerical_cols], y, odds


def get_league_data(league, seasons, season_test):
    # Read the data
    X_full = pd.read_csv(f'./leagues/{league}/formatted_data/{seasons}.csv', index_col=0)
    X_full.replace(' ', np.nan, inplace=True)
    X_full = X_full.dropna(subset=['home_odds', 'away_odds', 'draw_odds'], how='any')
    X_test_full = X_full[X_full['season'] == season_test]
    X_full = X_full[X_full['season'] < season_test]

    # Remove rows with missing target, separate target from predictors
    y = X_full.winner
    X_full.drop(['winner', 'home_score', 'away_score', 'home_odds',
                'away_odds', 'draw_odds'], axis=1, inplace=True)

    print(X_test_full[['date','season','home_team','away_team', 'winner']])
    X_test_full, y_test, odds_test = separate_dataset_info(X_test_full)

    return X_full, y, X_test_full, y_test, odds_test


def set_numerical_categorical_cols(X):
    # "Cardinality" means the number of unique values in a column
    # Select categorical columns with relatively low cardinality (convenient but arbitrary)
    categorical_cols = [cname for cname in X.columns if
                        (X[cname].nunique() < 10 and
                        X[cname].dtype in ['object'])]

    # Select numerical columns
    numerical_cols = [cname for cname in X.columns if
                      X[cname].dtype in ['float64']]

    # Select int columns
    int_cols = [cname for cname in X.columns if
                      X[cname].dtype in ['int64', 'int32']]

    return categorical_cols, numerical_cols, int_cols


def filter_datasets(X_full, y, X_test_full):
    # Keep selected columns only
    my_cols = lo.filtered_cols
    y_train = y.copy()
    X_train = X_full[my_cols]
    X_test = X_test_full[my_cols]

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

        X_train[f"Cluster_{i+1}"] = X_train[f"Cluster_{i+1}"].astype('int32')

        kmeans_scaler_list.append(kmeans_scaler)
        features_kmeans_list.append((features_to_explore[i:i+2], kmeans))

    return X_train, kmeans_scaler_list, features_kmeans_list

def apply_kmeans(X_test, kmeans_scaler_list, features_kmeans_list):
    for i, fk in enumerate(features_kmeans_list):
        features, kmeans = fk
        X_test[f"Cluster_{i+1}"] = kmeans.predict(scale_dataset(X_test.loc[:, features], kmeans_scaler_list[i], just_transform=True))
        X_test[f"Cluster_{i+1}"] = X_test[f"Cluster_{i+1}"].astype('int32')

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
    X_pca = X_pca.astype('int64')

    X_final = pd.concat([X_train, X_pca], axis=1)
    return X_final, features_to_explore, pca_scaler, pca

def apply_pca(X_test, scaler, pca, features_to_explore):
    X_pca = pca.transform(scale_dataset(X_test.loc[:, features_to_explore], scaler, just_transform=True))

    # Convert to df
    component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
    X_pca = pd.DataFrame(X_pca, columns=component_names, index=X_test.index)
    X_pca = X_pca.astype('int64')

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
            'score': None,
            'voting': False
        },
        'knn': {
            'estimator': KNeighborsClassifier(n_neighbors=40),
            'params': None,
            'score': None,
            'voting': True
        },
        'logistic_regression': {
            'estimator': LogisticRegression(random_state=0),
            'params': None,
            'score': None,
            'voting': False
        },
        'svm': {
            'estimator': SVC(
                probability=True,
                random_state=0
            ),
            'params': None,
            'score': None,
            'voting': False
        },
        'random_forest': {
            'estimator': RandomForestClassifier(random_state=0, n_estimators=750),
            'params': None,
            'score': None,
            'voting': True
        },
        'mlp': {
            'estimator': MLPClassifier(random_state=0),
            'params': None,
            'score': None,
            'voting': False
        },
        'neural_network': {
            'estimator': create_neural_network(),
            'params': None,
            'score': None,
            'voting': False
        },
    }

    voting_classifier_estimators = []
    for model in models_dict.keys():
        if models_dict[model]['voting'] and not isinstance(models_dict[model]['estimator'], Sequential): voting_classifier_estimators.append((model, models_dict[model]['estimator']))
    if voting_classifier_estimators: models_dict['voting_classifier'] = {'estimator': VotingClassifier(estimators=voting_classifier_estimators, voting='soft')}
        
    return models_dict

def build_pipeline(X_train, y_train, model, epochs=10, batch_size=32):
    categorical_cols, numerical_cols, int_cols = set_numerical_categorical_cols(X_train)

    # Preprocessing for numerical data
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())
    ])

    # Preprocessing for int data
    int_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())
    ])

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('int', int_transformer, int_cols),
            ('cat', categorical_transformer, categorical_cols),
        ])

    # Create a neural network model
    if isinstance(model, Sequential):
        # X_train_transformed_input = X_train
        X_train_transformed_input = preprocessor.fit_transform(X_train)
        build_neural_network(model, X_train_transformed_input)
        fit_kwargs = {'model__epochs': epochs, 'model__batch_size': batch_size}

        # Encode the target variable
        encoded_labels = encode_labels(y_train)
        y_train = encoded_labels
    else:
        fit_kwargs = {}

    # Bundle preprocessing and modeling code in a pipeline
    pipeline = Pipeline(steps=
                           [('preprocessor', preprocessor),
                            # ('feature_selection', SelectKBest(score_func=mutual_info_classif, k='all')),  # Select features based on mutual information
                            # ('pca', PCA(random_state=0), ),  # Perform PCA
                            # ('kmeans', KMeans(n_clusters=5, n_init=10, random_state=0)),  # Perform clustering
                            ('model', model)
                            ])

    # Preprocessing of training data, fit model
    pipeline.fit(X_train, y_train, **fit_kwargs)
    
    # Add the encoding and decoding functions to the pipeline object
    if isinstance(model, Sequential):
        pipeline.encode_labels = encode_labels
        pipeline.decode_labels = decode_labels
    
    return pipeline

def get_pred_odds(probs):
    return 1/probs

def get_bet_value(odds, probs, bankroll, strategy='kelly'):
    if strategy == 'kelly': 
        q = 1 - probs  # Probability of losing
        b = odds - 1  # Net odds received on the bet (including the stake)
        return ((bankroll * (probs * b - q)) / b) * 0.25
    elif strategy == 'bankroll_pct':
        return bankroll * 0.05

def get_bet_odds_probs(bet):
    if bet['pred'] == 'H': return bet['home_odds'], bet['home_probs']
    if bet['pred'] == 'A': return bet['away_odds'], bet['away_probs']
    if bet['pred'] == 'D': return bet['draw_odds'], bet['draw_probs']

def bet_worth_it(bet_worth, odds):
    # return True
    return bet_worth >= 5 and odds > 1.7
    
def get_bet_value_by_row(row, bankroll, strategy='kelly'):
    odds, probs = get_bet_odds_probs(row)
    return get_bet_value(odds, probs, bankroll, strategy)

def get_match_profit(row):
    odds, probs = get_bet_odds_probs(row)
    if not bet_worth_it(row['bet_worth'], odds): return 0
    if row['winner'] == row['pred']:
        return (odds*row['bet_worth']) - row['bet_worth']
    else:
        return -row['bet_worth']


def build_pred_df(my_pipeline, X_test, y_test, odds_test, bankroll=400, is_neural_net=False):
    if is_neural_net:
        test_probs = my_pipeline.predict(X_test)
        preds_test = my_pipeline.decode_labels(test_probs.argmax(axis=1))
        preds_test_labels = my_pipeline.decode_labels(preds_test)
        labels = np.unique(np.concatenate((y_test, preds_test_labels)))

    else:
        test_probs = my_pipeline.predict_proba(X_test)
        preds_test = my_pipeline.predict(X_test)
        labels = my_pipeline.classes_

    print('Classification Report:')
    report = classification_report(y_test, preds_test)
    print(report)

    print('Confusion Matrix:')
    matrix = confusion_matrix(y_test, preds_test, labels=labels)
    print(matrix)

    probs_test_df = pd.DataFrame(test_probs, index=y_test.index, columns=['away_probs', 'draw_probs', 'home_probs'])
    preds_test_df = pd.DataFrame(preds_test, index=y_test.index, columns=['pred'])
    test_results_df = pd.concat([y_test, preds_test_df, probs_test_df, odds_test], axis=1)

    print('\n')
    for l in labels:
        n_times = len(preds_test_df[preds_test_df['pred'] == l])
        print(f"Times when {l} was predicted: {n_times} ({round(n_times/len(preds_test_df), 2)})")

    test_results_df['progress'] = bankroll
    test_results_df['current_bankroll'] = bankroll

    for i, row in test_results_df.iterrows():
        odds, probs = get_bet_odds_probs(row)
        previous_bankroll = test_results_df.at[i-1, 'progress'] if i > 0 else bankroll
        bet_worth = get_bet_value(odds, probs, previous_bankroll, strategy=lo.strategy)
        test_results_df.at[i, 'bet_worth'] = bet_worth
        profit = get_match_profit(test_results_df.iloc[i])
        test_results_df.at[i, 'profit'] = profit
        test_results_df.at[i, 'progress'] = previous_bankroll + profit

    print('\nTotal bets:', len(test_results_df[test_results_df['profit'] != 0]))
    print('Model profit:', test_results_df.profit.sum())
    negative_consecutive_count = test_results_df['profit'].lt(0).astype(int).groupby((test_results_df['profit'] >= 0).cumsum()).sum().max()
    print('Maximum negative sequence: ', negative_consecutive_count)
    positive_consecutive_count = test_results_df['profit'].gt(0).astype(int).groupby((test_results_df['profit'] < 0).cumsum()).sum().max()
    print('Maximum positive sequence: ', positive_consecutive_count)
    print('Maximum bet worth:', test_results_df.bet_worth.max())
    print('Minimum bet worth:', test_results_df[test_results_df['profit'] != 0].bet_worth.min())

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
    return 1 if row['pred'] == row['winner'] else 0    

def simulate(X_train, y_train, X_test, y_test, odds_test, betting_starts_after_n_games, verbose=1):
    models_dict = get_models()

    # Only predicting after 15 team games. It's shown that is more profitable
    X_test_filtered = X_test.reset_index(drop=True)[betting_starts_after_n_games:]
    y_test_filtered = y_test.reset_index(drop=True)[betting_starts_after_n_games:]
    odds_test_filtered = odds_test.reset_index(drop=True)[betting_starts_after_n_games:]

    progress_data = []
    best_results = -9999
    best_results_df = None
    best_pipeline = None
    for model in models_dict.keys():
        print(f"\nResults for model {model}:")
        my_pipeline = build_pipeline(X_train, y_train, models_dict[model]['estimator'])
        if not len(X_test_filtered): continue

        is_neural_net = isinstance(models_dict[model]['estimator'], Sequential)
        test_results_df = build_pred_df(my_pipeline, X_test_filtered, y_test_filtered, odds_test_filtered, is_neural_net=is_neural_net)
        
        if verbose > 1: display(test_results_df)
        # if verbose > 1 or model == 'knn': plot_betting_progress(test_results_df)

        test_results_df['won'] = test_results_df.apply(lambda x: won_bet(x), axis=1)
        total_won = test_results_df[test_results_df['profit'] != 0]['won'].sum()
        progress_data.append([test_results_df['profit'].sum(), total_won/len(test_results_df[test_results_df['profit'] != 0])])

        # Define selected model for production
        if model == 'knn':
            best_results = test_results_df['profit'].sum()
            best_results_df = test_results_df
            best_pipeline = my_pipeline
        
    cols = ['profit', 'test_score']
    profit_df = pd.DataFrame(progress_data, columns=cols, index=models_dict.keys())
    if verbose > 0: display(profit_df)
    if verbose > 1: display(best_results_df.describe())

    # for i, row in best_results_df.iterrows():
    #     print(f"\n{row['home_team']} x {row['away_team']}: {row['pred']}/{row['winner']} {'WON' if row['won'] else ''}")
    #     print(f"Bankroll: {row['progress']}")
    #     print(f"Bet worth: {row['bet_worth']}")
    #     print(f"Profit: {row['profit']}")
    #     print(f"H{row['home_odds']} A{row['away_odds']} D{row['draw_odds']}")

    return best_pipeline

# Convert the labels to one-hot encoded vectors
def encode_labels(labels):
    encoded_labels = np.zeros((len(labels), 3))
    for i, label in enumerate(labels):
        if label == 'H':
            encoded_labels[i] = [1, 0, 0]
        elif label == 'D':
            encoded_labels[i] = [0, 1, 0]
        elif label == 'A':
            encoded_labels[i] = [0, 0, 1]
    return encoded_labels


# Decode the predictions
def decode_labels(predictions):
    decoded_labels = []
    for prediction in predictions:
        if np.argmax(prediction) == 0:
            decoded_labels.append('H')
        elif np.argmax(prediction) == 1:
            decoded_labels.append('D')
        elif np.argmax(prediction) == 2:
            decoded_labels.append('A')
    return decoded_labels

def create_neural_network():
    # Create a neural network model
    model = Sequential()
    return model

def build_neural_network(model, X_train):
    # Add layers to the model
    model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
