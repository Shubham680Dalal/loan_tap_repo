# Loan Tap Predictions
## Problem Statement

While LoanTap has a generally healthy loan portfolio with 80% of loans being fully paid, there remains room for improvement in reducing the 20% default rate to enhance profitability and reduce financial risk. Additionally, LoanTap faces challenges in optimizing loan approval processes and tailoring loan products to different customer segments and geographical areas to minimize defaults and maximize returns.

## Model Performance

Using a simple logistic regression model with stratified KFold cross-validation, SMOTE for handling class imbalance, and hyperparameter tuning on penalty ('L1', 'L2') and regularization parameter C, the best estimator was selected based on the Beta_F2_score (beta=2, favoring recall to minimize the approval of risky customers).

The test performance of the best estimator showed:
- Accuracy: 83%
- Recall: 65.4%
- Precision: 55.2%
- Beta_F2_score: 63% (at an optimal threshold of 0.45)

## Insights and Recommendations

- **Higher Loan Amounts:** Data shows that higher loan amounts are associated with a higher likelihood of default. LoanTap should consider adjusting loan amounts based on the borrower’s risk profile to mitigate this risk.
  
- **Geographical Influence:** Borrower pincode significantly influences loan default rates, indicating the need for geographical considerations in risk assessment and loan approval processes.
  - Implement stricter credit policies and lower loan limits for high-risk pincodes (11650, 86630).
  - Focus on safer regions (e.g., pincodes 48052, 70466, 30723) where the default risk is significantly lower.

- **Loan Terms and Categories:** Offer lower loan amounts and shorter terms to borrowers in higher-risk categories to mitigate potential losses.

- **Debt and Revolving Utilization:** Cap the debt-to-income ratio and revolving utilization rates to reduce the risk of over-leveraging.

- **Customer Grade:** Customers with 'A' grade ratings are significantly more likely to repay their loans fully, suggesting that focusing on acquiring and retaining high-grade customers can improve overall portfolio performance.

- **Homeownership Status:** Homeownership status (e.g., mortgage holders) correlates with lower default rates, indicating this as an important factor in credit assessments.

- **Profession-Based Risk:** Specific professions, such as Directors, Project Managers, Teachers, and Managers, show higher rates of full repayment, indicating lower risk. LoanTap could benefit from targeting and tailoring loan products to these professional segments.

## Proactive Risk Mitigation

- Develop early warning systems to monitor at-risk borrowers, particularly those with increasing DTI and revolving utilization metrics.
- Provide financial counseling and restructuring options for borrowers showing signs of financial distress.

## Data Pipeline for Extraction, ML Model, and API Deployment


### Flowchart Representation

```mermaid
flowchart TD
    A[Data Sources] --> B[DB2 (Appstream - AWS)]
    A --> C[Databricks]
    A --> D[Local Data Warehouse]
    B --> |Manual Extraction| E[Preprocessing (pandas)]
    C --> |SQL Queries| E
    D --> |SQL Queries| E
    E --> F[Merged Data Stored in Local Data Warehouse]
    F --> G[Basic EDA]
    G --> H[Create Virtual Environment & Repository]
    H --> I[Load Data & Train Model]
    I --> J[Data Preprocessing & Cleaning]
    I --> K[Model Training & Hyperparameter Tuning with MLflow]
    K --> L[MLflow Logging & UI]
    L --> M[Save Best Model as Pickle File]
    M --> N[Streamlit & Flask Dashboard]
    N --> O[Dockerize Application]
    O --> P[Push to AWS ECR]
    P --> Q[Create AWS ECS Task & Deploy Container]
    Q --> R[API Hosting & Deployment]
    Q --> T[CI/CD with Bitbucket/Jira]
Details
1. Data Extraction and Integration
DB2 (Appstream in AWS):

Since DB2 in Appstream doesn’t allow direct ODBC connections, data must be manually extracted. You can export data into CSV or other formats using DB2 tools or AWS services like S3.
Manual Step: After manual extraction, import the data into your local system and use pandas for processing:
python
Copy code
import pandas as pd

df_db2 = pd.read_csv('db2_data.csv')
Databricks: Use the JDBC/SQL connector for querying tables directly from Databricks.

python
Copy code
from databricks import sql
conn = sql.connect(server_hostname='databricks-server',
                   http_path='databricks-cluster',
                   access_token='your-token')
query = "SELECT * FROM business_events_table"
df_databricks = pd.read_sql(query, conn)
Local Data Warehouse: Connect using pyodbc or SQLAlchemy to perform SQL queries.

python
Copy code
import pyodbc

conn = pyodbc.connect('DSN=DataWarehouse;UID=user;PWD=password')
query = "SELECT * FROM warehouse_table"
df_warehouse = pd.read_sql(query, conn)
Data Merging: After extracting data from all sources, merge them using pandas.

python
Copy code
merged_df = pd.merge(df_db2, df_databricks, on='common_column')
final_df = pd.merge(merged_df, df_warehouse, on='common_column')
2. Data Preprocessing
Handle missing values, duplicates, and perform any necessary feature engineering.

python
Copy code
final_df.fillna(method='ffill', inplace=True)
final_df.drop_duplicates(inplace=True)
Store the cleaned and merged data back into your local data warehouse.

python
Copy code
final_df.to_sql('merged_table', conn, if_exists='replace', index=False)
3. EDA (Exploratory Data Analysis)
Perform basic exploratory analysis on the extracted data.
python
Copy code
import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(final_df)
plt.show()
4. Environment Setup and Repository Creation
Virtual Environment: Create a virtual environment and activate it.

bash
Copy code
python3 -m venv myenv
source myenv/bin/activate
Bitbucket/Jira Repository:

Create the necessary files for your project:

bash
Copy code
touch requirements.txt Dockerfile README.md
Add dependencies to requirements.txt:

bash
Copy code
pandas==1.4.2
scikit-learn==1.1.0
mlflow==1.26.1
streamlit==1.10.0
Create a Dockerfile:

Dockerfile
Copy code
FROM python:3.9-slim-buster
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app.py"]
5. Data Cleaning and Model Training
Clean and preprocess the data.
python
Copy code
from sklearn.model_selection import train_test_split

X = final_df.drop('target', axis=1)
y = final_df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
6. Model Training and Hyperparameter Tuning with MLflow
Use GridSearchCV for hyperparameter tuning.

python
Copy code
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20]}
grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid.fit(X_train, y_train)

mlflow.log_param("best_params", grid.best_params_)
Track experiments and log models using MLflow:

python
Copy code
import mlflow

mlflow.sklearn.log_model(grid.best_estimator_, "model")
7. Streamlit and Flask for MVP Dashboard
Create a Streamlit app for visualization and user interaction.
python
Copy code
import streamlit as st
import pickle

model = pickle.load(open('model.pkl', 'rb'))
user_input = st.text_input("Enter input:")
prediction = model.predict([user_input])
st.write(f"Prediction: {prediction}")
8. Dockerize Application
Create a Docker image for your app and push it to AWS ECR:
bash
Copy code
docker build -t myapp .
aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <account-id>.dkr.ecr.<region>.amazonaws.com
docker tag myapp:latest <account-id>.dkr.ecr.<region>.amazonaws.com/myapp:latest
docker push <account-id>.dkr.ecr.<region>.amazonaws.com/myapp:latest
9. AWS ECS Deployment
Create an ECS task definition that references the ECR image, and deploy your application:
In the AWS ECS Console, create a new task definition and configure the task with the image URL from ECR.
Set the desired number of tasks, configure autoscaling, and define the network settings.
10. CI/CD Pipeline with Bitbucket/Jira
CI/CD Setup: In Bitbucket, create a pipeline configuration to automate testing, building, and deploying your Docker image to AWS.

Add the following .bitbucket-pipelines.yml:
yaml
Copy code
image: python:3.9-slim-buster

pipelines:
  default:
    - step:
        name: Test, Build and Deploy
        script:
          - pip install -r requirements.txt
          - pytest
          - docker build -t myapp .
          - aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <account-id>.dkr.ecr.<region>.amazonaws.com
          - docker tag myapp:latest <account-id>.dkr.ecr.<region>.amazonaws.com/myapp:latest
          - docker push <account-id>.dkr.ecr.<region>.amazonaws.com/myapp:latest
Jira Integration: Track issues and deploys in Jira by linking Bitbucket repositories and using Jira issues to trigger specific builds and deployments.

vbnet
Copy code

This code block contains the flowchart, all necessary steps, and configuration files needed to complete your pipeline workflow, with references to Bitbucket, AWS ECS