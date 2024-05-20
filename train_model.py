import numpy as np


def get_folds(data, n_folds, observation_size):
    folds = []
    fold_size = len(data) // n_folds
    if observation_size > fold_size:
        raise ValueError("Observation size should be less than the fold size")
    for i in range(n_folds):
        snapshots = data[i * fold_size:i * fold_size + fold_size]
        observation_windows = []
        for j in range(len(snapshots) - observation_size + 1):
            observation_windows.append(snapshots[j:j + observation_size])
        folds.append(observation_windows)
    return folds


if __name__ == '__main__':
    import pandas as pd

    from feature_extraction import extract_train_data
    from get_data import aggregate_into_snapshots, get_hypertext, M
    from sklearn import tree
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score

    dataset = get_hypertext()
    delta_t = 20 * M
    n_folds = 5
    observation_size = 10

    # data = aggregate_into_snapshots(dataset, delta_t=delta_t)
    # folds = get_folds(data, n_folds, observation_size)
    # for i, fold in enumerate(folds):
    #     clf = tree.DecisionTreeClassifier()
    #     print(f"Fold {i + 1}")
    #     X_trains = []
    #     Y_trains = []
    #     for observation_window in fold[:-1]:
    #         X_train, Y_train = extract_train_data(observation_window[:-1], observation_window[-1])
    #         X_trains.append(X_train)
    #         Y_trains.append(Y_train)
    #     X_test, Y_test = extract_train_data(fold[-1][:-1], fold[-1][-1])
    #     X_test.to_csv(f'X_test_{i + 1}.csv', index=False)
    #     Y_test.to_csv(f'Y_test_{i + 1}.csv', index=False)
    #     pd.concat(X_trains).to_csv(f'X_train_{i + 1}.csv', index=False)
    #     pd.concat(Y_trains).to_csv(f'Y_train_{i + 1}.csv', index=False)

    for i in range(n_folds):
        X_train = pd.read_csv(f'train_test_data/X_train_{i + 1}.csv', dtype=np.float32)
        Y_train = pd.read_csv(f'train_test_data/Y_train_{i + 1}.csv', dtype=np.float32)
        X_test = pd.read_csv(f'train_test_data/X_test_{i + 1}.csv', dtype=np.float32)
        Y_test = pd.read_csv(f'train_test_data/Y_test_{i + 1}.csv', dtype=np.float32)

        X_train = X_train.reindex(sorted(X_train.columns), axis=1)
        X_test = X_test.reindex(sorted(X_test.columns), axis=1)

        # Check and replace inf/nan in X_train and X_test
        X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_test.replace([np.inf, -np.inf], np.nan, inplace=True)

        # You might want to fill NaN values if there are any
        X_train.fillna(X_train.mean(), inplace=True)
        X_test.fillna(X_test.mean(), inplace=True)

        clf = tree.DecisionTreeClassifier()
        clf.fit(X_train, Y_train)
        y_pred = clf.predict(X_test)

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
        results_df = pd.DataFrame({'True Values': Y_test.to_numpy()[:,0], 'Predicted Values': y_pred})
        print(results_df)
