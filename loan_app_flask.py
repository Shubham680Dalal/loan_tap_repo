from flask import Flask, request, jsonify
#FLASK_APP=loan_app_flask.py flask run
app=Flask(__name__)
print(__name__)

@app.route("/")
def hello_world():
    return "<p> Hello World!</p>"



import pandas as pd
import datetime
import preprocessing_func as pf
import pickle


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

@app.route("/predict",methods=['POST'])
def prediction():
    loan_req=request.get_json()

    df = pd.DataFrame(loan_req, index=[0])
    df,drop_str_cols=pf.narrow_transformations(df,non_collinear_columns,drop_str_cols=[])
    df=pf.wider_transformations(df,None,is_train=False,target_maps=target_maps,num_imputer=num_imputer,cat_imputer=cat_imputer,outliers=outliers)
    df=pf.transform_standardize_data(df,[col for col in numeric_cols if col in df],scalar)     
    #st.dataframe(df)
    ypred_val_prob=best_estimator.predict_proba(df)[:,1][0]
    ypred='Fully Paid' if ypred_val_prob<best_val_th else 'Charged Off'
    ypred_val_prob=1-ypred_val_prob if ypred_val_prob<best_val_th else ypred_val_prob
    return {"Loan Status":ypred,"Probability":ypred_val_prob}