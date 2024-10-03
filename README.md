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

@startuml
!define RECTANGLE class
!define DIAMOND diamond

RECTANGLE "Data Warehouse\n(Local System)" as DW
RECTANGLE "DB2 Database\n(Appstream)" as DB2
RECTANGLE "Databricks\n(Business Events)" as DB
RECTANGLE "Data Extraction\n (SQL Queries)" as DE
RECTANGLE "Data Preprocessing\n (Pandas)" as DP
RECTANGLE "New Tables\n (Data Warehouse)" as NT
RECTANGLE "Basic EDA\n (Pandas Profiling)" as EDA
RECTANGLE "Model Training\n (MLFlow)" as MT
RECTANGLE "Hyperparameter Tuning\n (GridSearchCV)" as HT
RECTANGLE "MLFlow UI\n (Hyperparameter Tracking)" as UI
RECTANGLE "Streamlit & Flask\n (API Creation)" as API
RECTANGLE "Docker Image\n (AWS ECR)" as DI
RECTANGLE "Containerized\n Application\n (AWS ECS)" as CA
RECTANGLE "CI/CD Pipeline\n (Bitbucket/Jira)" as CI

DIAMOND "Data Extraction\n (ODBC Connection)" as DEC
DIAMOND "Git Repository\n (Bitbucket)" as REPO
DIAMOND "Environment Setup\n (Docker)" as ENV

DE --> DEC : "ODBC/SQL Queries\n\n(Extract Data)"
DEC --> DW : "Extracted Data"
DEC --> DB2 : "Extracted Granular Info"
DEC --> DB : "Business Events Data"

DW --> DP : "Extracted Data"
DB2 --> DP : "Granular Info"
DB --> DP : "Business Events Data"

DP --> NT : "Processed Data\n\n(Create New Tables)"
NT --> EDA : "New Tables"

EDA --> MT : "Processed Data\n\n(Train/Test Split)"
MT --> HT : "Trained Models"

HT --> UI : "Best Estimators\n\n(Log & Save)"
UI --> API : "Model Evaluations\n(Visualize Results)"

API --> DI : "Create Docker Image"
DI --> CA : "Deploy in ECS"

CA --> CI : "Integrate CI/CD"

REPO --> ENV : "Create Repository\n (requirements.txt,\n README.md, Dockerfile)"
ENV --> MT : "Environment Setup"

@enduml
Key Components in the Workflow
Data Extraction: Connect to the Data Warehouse, DB2, and Databricks using ODBC connections to extract the necessary data.
Data Preprocessing: Use Pandas to clean and preprocess the data, merging data from different sources.
Basic EDA: Conduct exploratory data analysis to understand the data.
Model Training: Set up MLFlow for model training, hyperparameter tuning, and logging.
Streamlit & Flask: Create an MVP dashboard for predictions and visualizations.
Docker: Create a Docker image and deploy it to AWS ECR and ECS.
CI/CD Pipeline: Integrate the workflow with Bitbucket or Jira for continuous integration and delivery.
Caveats and Requirements
Caveats:
Data Access:

Ensure proper access permissions to the Data Warehouse, DB2, and Databricks.
You may need credentials and ODBC drivers installed on your local machine or server.
Environment Setup:

Conflicts may arise with package dependencies. Consider using virtual environments (e.g., venv or conda) to manage dependencies.
Ensure consistent Python versions between development and production environments.
Data Size:

Large datasets may lead to long processing times. Optimize SQL queries and data preprocessing steps to handle big data efficiently.
Resource Limits:

Be aware of resource limits (CPU, memory) in AWS ECS when deploying containers, especially if using large models or datasets.
Model Performance:

Monitor model performance in production. Consider setting up alerting mechanisms for any performance degradation.
Installation Requirements:
Libraries:

Python Packages: pandas, scikit-learn, mlflow, flask, streamlit, SQLAlchemy (for DB connections)
Docker: Ensure Docker is installed on your development and deployment environments.
Access Requests:

Request access to Bitbucket/Jira for repository creation and CI/CD setup.
Ensure you have permissions to create and manage AWS resources (ECR, ECS).
Data Visualization Tools:

Install necessary libraries for data visualization (e.g., matplotlib, seaborn).
Development Tools:

IDE (like PyCharm, VSCode) for code development and debugging.
Documentation:

Maintain clear documentation in the README.md file regarding setup, usage, and any dependencies.
Conclusion
This workflow provides a comprehensive overview of your MVP pipeline, addressing key steps from data extraction to deployment. By following this outline, you can streamline the development process and ensure efficient collaboration with your team. If you need further adjustments or have additional questions, feel free to ask!
