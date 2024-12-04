**A Data engineering and Machine Learning Approach to African Conflict Analysis:
	Predicting Fatalities and Classifying Event Types**

**Introduction**
This report presents a comprehensive data engineering and machine learning analysis of conflict events in Africa using the Armed Conflict Location and Event Data Project(ACLED) dataset( https://acleddata.com/data). The project has implemented a robust data pipeline and developed machine learning models to predict fatalities and classify event types, providing valuable insights for humanitarian organizations, policy makers, and other stakeholders.
Political violence and civil unrest in Africa pose significant challenges for local communities,  humanitarian organizations and policymakers. This project aims to tackle these challenges through two main objectives: predicting fatalities in conflict events and classifying the types of these events. To achieve this, two models were developed to provide actionable insights for decision-makers, utilizing a dataset of 388, 402 events spanning from 1997 to 2024.
Data Engineering Pipeline Architecture
The project implemented a data pipeline utilizing Amazon Web Services(AWS) infrastructure. Raw data ingestion begins with batch processing through Python scripts, storing the ACLED dataset in an S3 bucket (Data Lake). Data processing occurs within Docker containers running on EC2 instances, with Apache Airflow orchestrating the workflow through DAG implementation. This architecture ensures efficient data flow from ingestion through transformation to final analysis. All code and documentation are maintained in GitHub repository (https://github.com/tenageb
 

**Data Quality Assessment and Preprocessing**
The dataset underwent rigorous quality assessment and preprocessing. Columns with more than 50% missing values were excluded. This improved data quality while maintaining analytical integrity. Categorical features were encoded using appropriate techniques, and feature engineering was performed to enhance model performance.
**Descriptive Analysis Results**

![image](https://github.com/user-attachments/assets/d33cb98b-cd38-447e-9f17-aa97b539ed19)


 
Figure 1Descriptive Analysis of African conflict
The exploratory data analysis revealed several critical patterns in conflict events across Africa. The time series analysis figure shows a clear upward trend in conflict events from 2019 to 2023 in five countries particularly in Sudan and Cameroon. 
The geographic distribution visualization in the upper left panel shows conflict hotspots concentrated in specific regions. Eastern and North Africa show particularly high concentrations of events, while Central Africa displays more moderate levels of conflict activity. This spatial pattern highlights the uneven distribution of conflicts across the continent and suggests regional-specific factors influencing conflict occurrence.

**Machine Learning Results Analysis**
**1.	Random Forest Ensemble Performance**
Examining the visualization of the Random Forest ensemble performance,  we can observe the model’s behavior across different numbers of trees. The line graph shows performance metrics plotted against the number of trees (50, 100, 150, 200), with the optimal performance achieved at 150 trees. 
 ![image](https://github.com/user-attachments/assets/20967d45-69e9-426c-943b-150fd414c159)

Figure 2 ML using ZINB regression and Random Forest classification models
**2.	Confusion Matrix**
Addressing the problem statement about classifying various types of conflict events in Africa and overlapping event characteristics, the confusion on figure 2 above provides insights into the model’s classification performance.
The Matrix reveals strong classification accuracy for certain event types, particularly protest and riots which suggest these events have more distinct characteristics that make them easier to identify. For instance, the model correctly identified 17,410 protest events and 7,923 riot events, with relatively few misclassifications between these categories. The matrix also exposes significant classification challenges between battles and violence against civilians.
**3.	Zero-Inflated Negative Binomial (ZINB) Regression Analysis**
ZINB regression is a model specifically designed for counting data that has excess zeros and show overdispersion(variance greater than the mean). The ZINB model was specifically chosen for this project due to the unique characteristics of conflict fatality data. In conflict datasets, many events result in zero fatalities, creating what’s known as zero-inflation in the data distribution. Traditional regression models often perform poorly with such distributions because they don’t account for the excess zeros.
Figures of the ZINB regression results across different fatality ranges indicates the scatter plots and fitted lines reveal distinct patterns for each fatality category:
Model Performance by Fatality Range
 	RMSE	R2
High Fatality(> 100)	89.71	0.902
Medium Fatality (10 - 100)	14.77	0.799
Low Fatality (> 10)	1.41	0..561
		

For high-fatality events (> 100 fatality), the model shows very good predictive power with an R-squared of 0.902 and p-value < 0.0001. The RMSE of 89.71 is relatively high but expected given the large magnitude of these events.
Medium- fatality events(10 - 100) show good predictive ability with R-squared of 0.799. The lower RMSE of 14.77 indicates better precision in predictions compared to high-fatality events.
Low-fatality events(< 10) show poor predictive power with R-Squared of 0.561, notably lower than the other categories. The small RMSE reflects the lower magnitude of these events.
Recommendations and Future Work
•	Based on the project’s outcomes, several recommendations emerge for scaling and improvement:
•	Develop more sophisticated feature engineering techniques to capture complex conflict dynamics
•	Incorporate additional machine learning models to improve predication accuracy across different conflict scenarios.
**Conclusion**
This project successfully demonstrated the potential of data engineering and machine learning in analyzing and predicting conflict events in Africa. The accuracy in high fatality range prediction and event classification provides valuable tools for humanitarian organizations, policy makers and other stakeholders. The integration of modeling approaches Random Forest and ZINB regression provides a comprehensive framework for understanding conflict dynamics. The varying performance across different fatality ranges highlights the complexity of conflict prediction and the importance of tailored approaches for different scenarios.
Prerequisite to download the dataset from the source:

**You can follow these steps to build your data request.**
1.	Begin with the ACLED API’s base URL.
https://api.acleddata.com/
2.	Add the ACLED endpoint.
https://api.acleddata.com/acled/
3.	Add the response format.
https://api.acleddata.com/acled/read.csv
4.	Input your credentials.
https://api.acleddata.com/acled/read.csv?key=your_key&email=your_email
5.	Add query filters specifying the countries for which you would like to receive data.
https://api.acleddata.com/acled/read.csv?key=your_key&email=your_email&country=Georgia|Armenia|Azerbaijan
6.	Add a query filter to specify the year in which the events occurred.
https://api.acleddata.com/acled/read.csv?key=your_key&email=your_email&country=Georgia|Armenia|Azerbaijan&year=2021
7.	Next, specify which data columns you want to receive.
https://api.acleddata.com/acled/read.csv?key=your_key&email=your_email&country=Georgia|Armenia|Azerbaijan&year=2021&fields=event_id_cnty|event_date|event_type|country|fatalities


