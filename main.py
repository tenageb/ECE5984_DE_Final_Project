import os
from datetime import datetime
from EDA import perform_eda
from transformation import transform_data
from featureExtraction import feature_extract
from build_train_model import build_train


def main():
    print("\n=== Starting Analysis Pipeline ===")

    try:
        # Set up paths
        user_home = os.path.expanduser("~")
        output_dir = os.path.join(user_home, "Documents", "African_Conflict_Results")
        os.makedirs(output_dir, exist_ok=True)
        print(f"Using output directory: {output_dir}")

        # Load data
        data_path = r"C:\Users\tenag\Desktop\ECE5984_Project\Africa_1997-2024_Oct04.csv"
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at: {data_path}")

        print("1. Performing EDA...")
        raw_data = perform_eda(data_path)

        print("\n2. Transforming data...")
        transformed_data = transform_data(raw_data)

        print("\n3. Extracting features...")
        feature_data = feature_extract(transformed_data)

        print("\n4. Building and training models...")
        model_results = build_train(feature_data)

        # Save results
        print("\n=== Pipeline completed successfully! ===")
        print(f"Results saved in: {output_dir}")

    except Exception as e:
        print(f"\nError in pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    main()
