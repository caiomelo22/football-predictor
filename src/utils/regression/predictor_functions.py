import pandas as pd
import numpy as np
from . import totals_columns as tc
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

def separate_dataset_info(X, y_column):
    y = X[y_column]

    X.drop(['winner', 'home_score', 'away_score',
                     'home_odds', 'away_odds', 'draw_odds'], axis=1, inplace=True)
    
    numerical_cols = [cname for cname in X.columns if
                      X[cname].dtype in ['float64']]
    return X[numerical_cols], y

def get_league_data(league, seasons, season_test, y_column):
    # Read the data
    X_full = pd.read_csv(f'./leagues/{league}/formatted_data/{seasons}.csv', index_col=0)
    X_full.replace(' ', np.nan, inplace=True)
    X_full = X_full.dropna(subset=['home_odds', 'away_odds', 'draw_odds'], how='any')
    X_test_full = X_full[X_full['season'] == season_test]
    X_full = X_full[X_full['season'] < season_test]

    # Remove rows with missing target, separate target from predictors
    y = X_full[y_column]
    X_full.drop(['winner', 'home_score', 'away_score', 'home_odds',
                'away_odds', 'draw_odds'], axis=1, inplace=True)

    X_test_full, y_test = separate_dataset_info(X_test_full, y_column)

    columns = tc.filtered_cols
    ha_prefix = y_column.split('_')[0]
    specific_ha_columns = [c for c in columns if (c.startswith(ha_prefix) and '_opp_' not in c) or (not c.startswith(ha_prefix) and '_opp_' in c)]

    X_full = X_full[specific_ha_columns]
    X_test_full = X_test_full[specific_ha_columns]

    return X_full, y, X_test_full, y_test

def regression_model_evaluation(X_train, y_train, X_test, y_test, pred_col):
    # Define a dictionary to store the models and their MSE scores
    models = {
        "Ridge Regression": Ridge(),
        "Random Forest": RandomForestRegressor(random_state=0, n_estimators=500),
        "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=50)
    }

    mse_scores = {}  # To store the MSE scores
    predictions = {}  # To store the predictions of each model

    for model_name, model in models.items():
        # Train the regression model using the training data
        model.fit(X_train, y_train)

        # Predict the target variables for the training and test datasets
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Calculate the MSE for the test dataset
        mse_scores[model_name] = mean_squared_error(y_test, y_test_pred)

        # Store the predictions of the current model
        predictions[model_name] = y_test_pred

    # Print the MSE scores for each model
    print(f'\nPredictions for {pred_col}:')
    for model_name, mse_score in mse_scores.items():
        print(f"{model_name} MSE: {mse_score}")

    # Get the best model based on the lowest MSE score
    best_model = min(mse_scores, key=mse_scores.get)

    # Create a dataframe to show the predicted values and actual values of the best model
    df_predictions = pd.DataFrame({
        f"Actual {pred_col}": y_test,
        f"Pred {pred_col}": predictions[best_model],
        f"Pred {pred_col} Rounded": predictions[best_model].round(0),
        f"Pred {pred_col} Rounded": predictions[best_model].round(0),
    })
    df_predictions[f'{pred_col} Difference'] = df_predictions[f'Pred {pred_col}'] - df_predictions[f'Actual {pred_col}']

    # Return the dictionary of MSE scores
    return df_predictions, models[best_model]