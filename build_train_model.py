from operator import index

import pandas as pd
import numpy as np
from numpy.ma.extras import average
from pandas.core.common import random_state
from scipy.special import y_pred
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import pickle
import s3fs

DIR = 's3:...Final_Data_Warehouse'

def build_train():
    """ Main function to train RF and ZINB models"""
    s3 = s3fs.S3FileSystem(anon=False)
    # Load classification data
    classification_data = {}
    input_file_names = {
        'X_train': 'classification_X_train.pkl',
        'X_test': 'classification_X_test.pkl',
        'y_train': 'classification_y_train.pkl',
        'y_test': 'classification_y_test.pkl'
    }

    for key, filename in input_file_names.items():
        with s3.open(f'{DIR}/{filename}', 'rb') as f:
            classification_data[key] = pickle.load(f)

    # Train Random Forest models
    rf_results = train_rf_ensemble(classification_data)

    # Load ZINB data from each range
    zinb_data = {}
    for range_type in ['low', 'medium', 'high']:
        try:
            zinb_data[range_type] = {
                'X_train': pickle.load(s3.open(f'{DIR}/zinb_{range_type}_X_train.pkl', 'rb')),
                'X_test': pickle.load(s3.open(f'{DIR}/zinb_{range_type}_X_test.pkl', 'rb')),
                'y_train': pickle.load(s3.open(f'{DIR}/zinb_{range_type}_y_train.pkl', 'rb')),
                'y_test': pickle.load(s3.open(f'{DIR}/zinb_{range_type}_y_test.pkl', 'rb'))
            }
            print(f"Successfully loaded{range_type} range data")
        except Exception as e:
            print(f'Error loading {range_type} range data: {str(e)}')
            continue

    # Train ZINB models
    zinb_results = train_zinb_models(zinb_data)

    return {
        'rf_metrics': rf_results["metrics"],
        'rf_fold_results': rf_results["fold_results"],
        'rf_confusion_matrices' : rf_results["confusion_matrices"],
        'zinb_metrics': zinb_results["metrics"]
    }

def train_rf_ensemble(data):
    s3 = s3fs.S3FileSystem(anon=False)
    n_trees_options = [50, 100, 150, 200]
    all_metrics = {}
    fold_results = []
    confusion_matrices = []

    for n_tress in n_trees_options:
        # Initialize RF model
        rf_model = RandomForestClassifier(
            n_estimators=n_tress,
            max_depth=10,
            random_state=42

        )
        # K-fold cross validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for fold,(train_idx, val_idx) in enumerate(kf.split(data["X_train"])):
            X_fold_train = data["X_train"].iloc[train_idx]
            y_fold_train = data["y_train"].iloc[train_idx]
            X_fold_val = data["X_train"].iloc[val_idx]
            y_fold_val = data["y_train"].iloc[val_idx]

            # Train and evaluate on fold
            rf_model.fit(X_fold_train, y_fold_train)
            fold_accuracy = rf_model.score(X_fold_val, y_fold_val)

            # Store fold results
            fold_results.append({
                'n_tress': n_tress,
                'fold': fold + 1,
                'accuracy': fold_accuracy
            })

        # Train final model on full training set
        rf_model.fit(data["X_train"], data["y_train"])
        y_pred = rf_model.predict(data["X_test"])
        y_prob = rf_model.predict_proba(data["X_test"])[:, 1]

        # Calculate confusion matrix
        cm = confusion_matrix(data["y_test"], y_pred)
        confusion_matrices.append({
            'n_trees': n_tress,
            'tn': cm[0 ,0],
            'fp': cm[0, 1],
            'fn': cm[1, 0],
            'tp': cm[1, 1]
        })

        # Calculate metrics
        metrics = {
            "n_trees": n_tress,
            "accuracy": float(rf_model.score(data["X_test"], data["y_test"])),
            "precision": float(precision_score(data["y_test"], y_pred, average="macro")),
            "recall": float(recall_score(data["y_test"], y_pred, average="macro")),
            "f1_score": float(f1_score(data["y_test"], y_pred, average="macro")),
        }
        all_metrics[str(n_tress)] = metrics

     # Save model
    model_path = f'{DIR}/rf_model_{n_tress}_trees.h5'
    with s3.open(model_path, 'wb') as f:
         pickle.dump(rf_model, f)
     # Save all results
    with s3.open(f'{DIR}/rf_ensemble_metrics.csv', 'w') as f:
        pd.DataFrame(all_metrics).T.to_csv(f)

    with s3.open(f'{DIR}/rf_fold_results.csv', 'w') as f:
        pd.DataFrame(fold_results).to_csv(f, index=False)

    with s3.open(f'{DIR}/rf_confusion_matrices.csv', 'w') as f:
        pd.DataFrame(confusion_matrices).to_csv(f, index=False)

    return {
        "metrics": all_metrics,
        "fold_results": fold_results,
        "confusion_matrices": confusion_matrices
    }

def train_zinb_models(zinb_data):
    """ Train ZINB models for different fatality ranges(low, medium and high)"""
    s3 = s3fs.S3FileSystem(anon=False)
    metrics = {}

    for range_type, data in zinb_data.items():
        if len(data["y_train"]) > 0:
            model = XGBRegressor(n_estimators = 100, learning_rate=0.1, max_depth=6, random_state=42)
            y_train_log = np.log1p(data["y_train"])
            model.fit(data["X_train"], y_train_log)

            y_pred_log = model.predict(data["X_test"])
            y_pred = np.expm1(y_pred_log)

            mse = float(np.mean((data["y_test"] - y_pred) ** 2))
            rmse = float(np.sqrt(mse))
            r2 = float(np.corrcoef(data["y_test"], y_pred)[0, 1] ** 2)

            metrics[range_type] = {"mse": mse, "rmse": rmse, "r2":r2}

            # Save metrics and model
            with s3.open(f'{DIR}/{range_type}_zinb_metrics.csv', 'w') as f:
                pd.DataFrame([metrics[range_type]]).to_csv(f, index=False)

            with s3.open(f'{DIR}/{range_type}_zinb_model.h5',  'wb') as f:
               pickle.dump(model, f)
    return {"metrics": metrics}

if __name__ == "__main__":
    build_train()

