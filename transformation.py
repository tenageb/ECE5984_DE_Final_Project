import pandas as pd
import s3fs

def transform_data():
    """ Load raw data from Data Lake, transform it and save the cleaned data to the Data Warehouse """
    try:
        # Initialize S3 files systems
        s3 = s3fs.S3FileSystem(anon=False)
        with s3.open(path, 'rb') as f:
            raw_data = pd.read_pickle(f)
            # Columns to drop
            columns_to_drop = [
                'assoc_actor_1','actor2', 'assoc_actor_2', 'inter2','admin1','admin2','admin2',
                'notes', 'tags', 'civilian_targeting','event_id_cnty', 'geo_precision', 'iso', 'interaction'
            ]
        # Transformation
        transformed_data =  raw_data.drop(columns = columns_to_drop, errors = 'ignore')
        transformed_data = transformed_data.dropna().drop_duplicates()
        transformed_data["event_date"] = pd.to_datetime(transformed_data["event_date"])
        transformed_data["month"] = transform_data["event_date"].dt.dayofweek

        # Save transformed data to Data Warehouse
        with s3.open(DIR, 'wb') as f:
            pd.to_pickle( transformed_data, f)
            return f"Data transformed and saved to Data Warehouse: {DIR}"
    except Exception as e:
        raise RuntimeError(f"Failed to transform data:{str(e)}")

if __name__ == "__main__":
    path = 's3://...Final_Data_Lake/raw_data.pkl'
    DIR = 's3://...Final_Data_Warehouse'
    transform_data()

