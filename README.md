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
