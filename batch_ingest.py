import pandas as pd
import pickle
import s3fs

def ingest_data():

        s3 = s3fs.S3FileSystem(anon=False)
        s3_csv = "s3://...Africa_1997-2024_Oct04.csv"
        path = "s3://...Final_Data_Lake"

        # Read CSV directly from S#
        with s3.open(s3_csv, 'rb') as f:
            data = pd.read_csv(f)

            # Save processed data back to S3
            with s3.open(path, 'wb') as f:
                pickle.dump(data, f)
            print( f"Success! Processed data shape : {data.shape}")
            return "Data ingestion completed successfully"



if __name__ == "__main__":
    ingest_data()

