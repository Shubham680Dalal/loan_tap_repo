import streamlit as st
import pandas as pd
import datetime
import preprocessing_func as pf
import pickle
import os

param_dict=pickle.load(open('best_estimator1.pkl','rb'))
param_grid=param_dict['param_grid']
checkpoint_index=param_dict['checkpoint_index']  ##one already covered
best_param_index=param_dict['best_param_index']
best_estimator=param_dict['best_estimator']

best_val_accuracy=param_dict['best_val_accuracy']
best_val_f_betascore=param_dict['best_val_f_betascore']
best_val_recall=param_dict['best_val_recall']
best_val_precision=param_dict['best_val_precision']
best_val_th=param_dict['best_val_th']
best_eq=param_dict['equation']
target_maps=param_dict['target_maps']
num_imputer=param_dict['num_imputer']
cat_imputer=param_dict['cat_imputer']
outliers=param_dict['outliers']
numeric_cols=param_dict['numeric_cols']
scalar=param_dict['scalar']
apply_vif=param_dict['apply_vif']
non_collinear_columns=param_dict['non_collinear_columns']

# Create a form
with st.form(key='loan_form'):
    st.write("# Loan Approval Prediction")
    st.write("### Main Features")
    # First row
    col1, col2, col3 = st.columns(3)

    with col1:
        loan_amnt = float(st.text_input("Loan Amount", 100))
        int_rate = float(st.text_input("Interest Rate", 8))
        dti = float(st.slider("DTI", 0.0, 25.0, step=0.5))

    with col2:
        term = st.selectbox("Term", ("36 months", "60 months"))
        installment = float(st.slider("Installment", 0, 2000, step=1))
        grade = st.selectbox("Grade", ("A", "B", "C", "D", "E", "F", "G"))

    with col3:
        sub_grade = st.selectbox("Sub Grade", ('B4', 'B5', 'B3', 'A2', 'C5', 'C3', 'A1', 'B2', 'C1', 'A5', 'E4',
                                               'A4', 'A3', 'D1', 'C2', 'B1', 'D3', 'D5', 'D2', 'E1', 'E2', 'E5',
                                               'F4', 'E3', 'D4', 'G1', 'F5', 'G2', 'C4', 'F1', 'F3', 'G5', 'G4',
                                               'F2', 'G3'))

    # Second row
    col4, col5, col6 = st.columns(3)

    with col4:
        emp_length = st.selectbox("Employment Length", ('10+ years', '4 years', '< 1 year', '6 years', '9 years',
                                                        '2 years', '3 years', '8 years', '7 years', '5 years', '1 year'))
        home_ownership = st.selectbox("Home Ownership", ('RENT', 'MORTGAGE', 'OWN', 'OTHER', 'NONE', 'ANY'))
        purpose = st.selectbox("Purpose", ('vacation', 'debt_consolidation', 'credit_card', 'home_improvement',
                                           'small_business', 'major_purchase', 'other', 'medical', 'wedding', 'car',
                                           'moving', 'house', 'educational', 'renewable_energy'))

    with col5:
        initial_list_status = st.selectbox("Initial List Status", ('w', 'f'))
        application_type = st.selectbox("Application Type", ('INDIVIDUAL', 'JOINT', 'DIRECT_PAY'))
        verification_status = st.selectbox("Verification Status", ('Not Verified', 'Source Verified', 'Verified'))

    with col6:
        pincode = st.selectbox("Pincode", ('22690', '05113', '00813', '11650', '30723', '70466', '29597', '48052',
                                           '86630', '93700'))
        state = st.selectbox("State", ('NH', 'MI', 'OR', 'AZ', 'PA', 'WY', 'NV', 'VA', 'SD', 'AP', 'SC',
       'MA', 'AL', 'MN', 'CT', 'LA', 'IL', 'DE', 'AA', 'NE', 'WV', 'WA',
       'OH', 'AK', 'ME', 'NJ', 'ID', 'KY', 'UT', 'CA', 'AE', 'RI', 'MS',
       'HI', 'ND', 'TX', 'MT', 'CO', 'NM', 'NY', 'NC', 'KS', 'WI', 'DC',
       'VT', 'IA', 'GA', 'MD', 'TN', 'AR', 'MO', 'OK', 'FL', 'IN'))
        annual_inc = float(st.text_input("Annual Income", 100))
        issue_d = st.date_input('Issue Date', value=datetime.date(2015, 1, 7)).strftime('%b-%Y')
        earliest_cr_line = st.date_input('Earliest Credit Line', value=datetime.date(1990, 1, 7)).strftime('%b-%Y')

    # Third row for additional float features
    st.write("### Additional Features")

    col7, col8 = st.columns(2)

    with col7:
        open_acc = int(st.text_input("Open Accounts", 0))
        revol_bal = int(st.text_input("Revolving Balance", 0))
        total_acc = int(st.text_input("Total Accounts", 0))

    with col8:
        pub_rec = int(st.text_input("Public Record", 0))
        revol_util = int(st.text_input("Revolving Utilization", 0))
        mort_acc = int(st.text_input("Mortgage Accounts", 0))
        pub_rec_bankruptcies = int(st.text_input("Public Record Bankruptcies", 0))

    # Submit button
    submit_button = st.form_submit_button(label="Submit")

# After the submit button is pressed
if submit_button:
    # Create a dictionary with all the entered values
    data = {
        'loan_amnt': [loan_amnt],
        'term': [term],
        'int_rate': [int_rate],
        'installment': [installment],
        'dti': [dti],
        'grade': [grade],
        'sub_grade': [sub_grade],
        'emp_length': [emp_length],
        'home_ownership': [home_ownership],
        'purpose': [purpose],
        'initial_list_status': [initial_list_status],
        'application_type': [application_type],
        'pincode': [pincode],
        'state':[state],
        'annual_inc': [annual_inc],
        'verification_status': [verification_status],
        'issue_d': [issue_d],
        'earliest_cr_line': [earliest_cr_line],
        'open_acc': [open_acc],
        'pub_rec': [pub_rec],
        'revol_bal': [revol_bal],
        'revol_util': [revol_util],
        'total_acc': [total_acc],
        'mort_acc': [mort_acc],
        'pub_rec_bankruptcies': [pub_rec_bankruptcies]
    }

    # Create a DataFrame from the dictionary
    
    df = pd.DataFrame(data)

    # Display the DataFrame
    st.write("### Output Based on Submitted Data")
    st.write(str({d:data[d][0] for d in data.keys()}))
    df,drop_str_cols=pf.narrow_transformations(df,non_collinear_columns,drop_str_cols=[])
    df=pf.wider_transformations(df,None,is_train=False,target_maps=target_maps,num_imputer=num_imputer,cat_imputer=cat_imputer,outliers=outliers)
    df=pf.transform_standardize_data(df,[col for col in numeric_cols if col in df],scalar)     
    #st.dataframe(df)
    ypred_val_prob=best_estimator.predict_proba(df)[:,1][0]
    ypred='Fully Paid' if ypred_val_prob<best_val_th else 'Charged Off'
    ypred_val_prob=1-ypred_val_prob if ypred_val_prob<best_val_th else ypred_val_prob
    st.write("Loan would be {}. Probability: {}".format(ypred,ypred_val_prob))
    
