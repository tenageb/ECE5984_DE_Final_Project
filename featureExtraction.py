import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import s3fs
import pickle

def feature_extract():
    s3 = s3fs.S3FileSystem(anon = False)
    DIR = 's3://...Final_Data_Warehouse'
    # Load cleaned data from S3 data warehouse
    clean_data_path = f"{DIR}/clean_data.pkl"
    with s3.open(clean_data_path, 'rb') as f:
        transformed_data = pd.read_pickle(f)

    # Create binary target
    transformed_data['has_fatalities'] = (transformed_data['fatalities'] > 0).astype(int)
    # Feature groups
    classification_features = ['event_type', 'actor1', 'country', 'region', 'time_precision',
                               'month', 'day_of_week']

    zinb_features = ['event_type', 'actor1', 'country', 'region', 'time_precision',
                               'month', 'day_of_week']
    # Add year for temporal trends
    transformed_data['year'] = pd.to_datetime(transformed_data['event_date']).dt.year

    # Create fatality range datasets
    low = transformed_data['fatalities'] < 10
    medium = (transformed_data['fatalities'] >=10) & (transformed_data['fatalities'] < 100)
    high = (transformed_data['fatalities'] >= 100)

    # Prepare datasets
    X_class = transformed_data[classification_features].copy()
    y_class = transformed_data['has_fatalities']

    X_zinb = transformed_data[zinb_features].copy()
    y_zinb = transformed_data['fatalities']

    # Encode categorical variables
    le_dict = {}
    categorical_features = [col for col in X_zinb.columns if X_zinb[col].dtype == 'object']

    for col in categorical_features:
        le_dict[col] = LabelEncoder()
        if col in X_zinb.columns:
                X_zinb[col] = le_dict[col].fit_transform(X_zinb[col])
        if col in X_class.columns:
                X_class[col] = le_dict[col].transform(X_class[col])


    # Scale numerical features
    scaler = StandardScaler()
    numeric_features = ['year', 'month', 'day_of_week']
    X_zinb[numeric_features] = scaler.fit_transform(X_zinb[numeric_features])

    # Split classification data
    X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
        X_class, y_class, test_size=0.2, random_state=42, stratify = y_class
    )

    # Split ZINB data for each range
    zinb_splits = {}
    for range_name, mask in [('low', low), ('medium', medium), ('high', high)]:
        X_range = X_zinb[mask]
        y_range = y_zinb[mask]

        if len(X_range) > 0:
            X_train, X_test, y_train, y_test = train_test_split(
                X_range, y_range, test_size = 0.2, random_state = 42
            )
            zinb_splits[range_name] = {
                'X_train' : X_train, 'y_test': y_test
            }

    # Save all datasets
    DIR = 's3://...Final_Data_Warehouse'

    # Save classification splits
    datasets = {
        'classification_X_train.pkl': X_train_class,
        'classification_X_test.pkl': X_test_class,
        'classification_y_train.pkl': y_train_class,
        'classification_y_test.pkl': y_test_class
    }

    # Add ZINB splits
    for range_name, splits_data in zinb_splits.items():
        for split_type, data in splits_data.items():
            datasets[f'zinb_{range_name}_{split_type}.pkl'] = data

    # Save each dataset
    for filename, data in datasets.items():
        with s3.open(f'{DIR}/{filename}', 'wb') as f:
            pickle.dump(data, f)

if __name__ =="__main__":
    feature_extract()


