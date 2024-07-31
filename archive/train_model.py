import numpy as np
from utils import *





if __name__ == '__main__':
    import pandas as pd

    from sklearn import tree, __all__, ensemble
    from sklearn import linear_model
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score



    for i in range(5):
        X_train = pd.read_csv(f'train_test_data/X_train_{i + 1}.csv', dtype=np.float32)
        Y_train = pd.read_csv(f'train_test_data/Y_train_{i + 1}.csv', dtype=np.float32)
        X_test = pd.read_csv(f'train_test_data/X_test_{i + 1}.csv', dtype=np.float32)
        Y_test = pd.read_csv(f'train_test_data/Y_test_{i + 1}.csv', dtype=np.float32)

        X_train = X_train.reindex(sorted(X_train.columns), axis=1)
        X_test = X_test.reindex(sorted(X_test.columns), axis=1)

        # Check and replace inf/nan in X_train and X_test
        X_train.replace([np.inf, -np.inf, np.NaN], np.nan, inplace=True)
        X_test.replace([np.inf, -np.inf, np.NaN], np.nan, inplace=True)

        X_train.fillna(0.0, inplace=True)
        X_test.fillna(0.0, inplace=True)

        clf = ensemble.RandomForestRegressor(n_estimators=10, random_state=42, max_depth=15, n_jobs=-1)
        clf.fit(X_train, Y_train.to_numpy().ravel())
        y_pred = clf.predict(X_test)
        # Extract the feature importances
        importances = clf.feature_importances_

        # Create a DataFrame to hold feature names and their corresponding importances
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': importances
        })

        # Sort features by importance
        feature_importance = feature_importance.sort_values(by='importance', ascending=False)

        # Display the most important features
        print(feature_importance)

        # Calculate metrics
        mae = mean_absolute_error(Y_test, y_pred)
        mse = mean_squared_error(Y_test, y_pred)
        rmse = np.sqrt(mse)  # Root Mean Squared Error
        r2 = r2_score(Y_test, y_pred)
        explained_variance = explained_variance_score(Y_test, y_pred)

        # Print metrics
        print(f"Mean Absolute Error: {mae}")
        print(f"Mean Squared Error: {mse}")
        print(f"Root Mean Squared Error: {rmse}")
        print(f"R-squared: {r2}")
        print(f"Explained Variance Score: {explained_variance}")

        # Create a DataFrame with true values and predicted values
        results_df = pd.DataFrame({'True Values': Y_test.to_numpy()[:, 0], 'Predicted Values': y_pred})
        print(results_df)
