from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import utils.predictor_functions as pf

league = 'premier-league'
seasons = '2017-2023'
season_test = 2022

# Read the data
X_full, y, X_test_full, y_test, odds_test = pf.get_league_data(league, seasons, season_test)

# Define categorical and numerical columns
categorical_cols, numerical_cols = pf.set_numerical_categorical_cols(X_full)

# Keep selected columns only
X_train, y_train, X_test = pf.filter_datasets(X_full, y, X_test_full, categorical_cols, numerical_cols)

# Transform numerical and categorical cols
X_train = pf.transform_x(X_train, categorical_cols, numerical_cols)
X_test = pf.transform_x(X_test, categorical_cols, numerical_cols)

# Get mi scores
first_mi_scores = pf.make_mi_scores(X_train, y_train)

# Create cluster features
pf.create_cluster_features(X_train, X_test, first_mi_scores)

# Get mi scores including the cluster features
second_mi_scores = pf.make_mi_scores(X_train, y_train)

# Create PCA features
X_train, X_test = pf.apply_pca_datasets(X_train, X_test, second_mi_scores)

# Get mi scores including the PCA features
third_mi_scores = pf.make_mi_scores(X_train, y_train)

# Get random forest model via random search
model = LogisticRegression(random_state=0, penalty='l2', C=0.1)
# model = RandomForestClassifier(criterion='entropy', max_depth=None, max_features='log2',
#                          min_samples_leaf=2, min_samples_split=25,
#                          n_estimators=500, random_state=0)

# Build pipeline
my_pipeline = pf.build_pipeline(X_train, y_train, model)

# Generate statistics
test_results_df = pf.build_pred_df(my_pipeline, X_test, y_test, odds_test)
