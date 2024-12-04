import pandas as pd
import matplotlib.pyplot as plt
import s3fs


def perform_eda():
    s3 = s3fs.S3FileSystem(anon=False)

    # Load data from Data lake
    print("Loading data from Data Lake")
    with s3.open(path, 'rb') as f:
        raw_data = pd.read_pickle(f)

    # Basic dataset information
    print("Generating dataset overview")
    dataset_summary = {
        "Total Records": len(raw_data),
        "Number of Features": len(raw_data.columns),
        "Shape": raw_data.shape,
    }
    print(dataset_summary)

    # Save dataset summary to Data Warehouse
    summary_path = f"{DIR}/dataset_summary.csv"
    with s3.open(summary_path, 'w') as f:
        pd.DataFrame([dataset_summary]).to_csv(f, index = False)

    # Generate and save visualizations to Data Warehouse
    print("Generating visualizations")
    plt.figure(figsize = (15, 6))
    raw_data["event_type"].value_counts().plot(kind = 'bar')
    plt.title("Event Types Distribution")
    plt.xticks(rotation = 45)
    plt.tight_layout()
    with s3.open(f"{DIR}/event_types.png", 'wb') as f:
        plt.savefig(f)
    plt.close()

    plt.figure(figsize=(15, 6))
    raw_data['country'].value_counts().head(10).plot(kind = 'bar')
    plt.title("Top 10 Countries by Number of Events")
    plt.xticks(rotation = 45)
    plt.tight_layout()
    with s3.open(f"{DIR}/top_countries.png", 'wb') as f:
        plt.savefig(f)
    plt.close()

    # Time series analysis
    raw_data['event_date'] = pd.to_datetime(raw_data["event_date"])
    monthly_events = raw_data.groupby(raw_data["event_date"].dt.to_period('ME')).size()
    plt.figure(figsize = (15, 6))
    plt.plot(monthly_events.index.astype(str), monthly_events.values, label = "All Events")
    plt.title("Monthly Conflict Trend(1997 - 2024)")
    plt.xlabel("Date")
    plt.ylabel("Number of Events")
    plt.grid(True, linestyle = "--", alpha = 0.7)
    plt.legend()
    plt.tight_layout()
    with s3.open(f"{DIR}/monthly_trends.png", 'wb') as f:
        plt.savefig(f)
    plt.close()

    # Event Distribution by Country
    top_5_countries = raw_data["country"].value_counts().head(5).index
    event_distribution = pd.crosstab(
        raw_data[raw_data["country"].isin(top_5_countries)]["country"],
        raw_data[raw_data["country"].isin(top_5_countries)]["event_type"],
    )
    event_distribution.plot(kind = 'bar', stacked = True, figsize = (15, 8))
    plt.title("Event Type Distribution in Top 5 Countries")
    plt.xlabel("Country")
    plt.ylabel("Number of Events")
    plt.legend(title = "Event Type", bbox_to_anchor = (1.05, 1), loc = "upper left")
    plt.tight_layout()
    with s3.open(f"{DIR}/event_distribution_by_country.png", 'wb') as f:
        plt.savefig(f)
    plt.close()

    print("EDA completed. Results saved to {DIR}")
    return "EDA process completed successfully"

if __name__ == "__main__":
    path = 's3://...Final_Data_Lake'
    DIR = 's3://...Final_Data_Warehouse'
    perform_eda()

