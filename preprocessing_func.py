import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from datetime import datetime,timedelta
import warnings
warnings.filterwarnings('ignore')
import time
import re
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer


pd.set_option('display.max_columns', 500)

## narrow transform -- dates (extracting quarter from here)
def convert_dates(df,col):
    data=df.copy()
    dates=data[col].str.split('-',expand=True)
    dates.columns=['mon','year']

    map_months=pd.date_range('2000-01-01','2000-12-31',freq='MS').strftime("%b").tolist()
    map_months={i[1]:i[0]+1 for i in enumerate(map_months)}
    
    dates['mon']=pd.to_numeric(dates.mon.map(map_months),errors='coerce')
    dates['year']=pd.to_numeric(dates['year'],errors='coerce')
    dates.columns=[col+'_mon',col+'_year']
    dates[col+'_Q']=(1+((dates[col+'_mon']-1)//3)).astype(str)
    
    data.drop([col],axis=1,inplace=True)
    data=pd.concat([data,dates[[col+'_Q']]],axis=1)
    return data

## One hot encoding
def dummy_str_col(d,cols):
    data=d.copy()
    drop_str_cols=[]
    str_cols=cols
    for str_col in str_cols:
        str_data=pd.get_dummies(data.loc[:,str_col]).astype(int)
        str_data.columns=[str_col+'_'+str(col) for col in str_data.columns]
        drop_str_cols.append(str_data.columns[-1])
        str_data.drop(str_data.columns[-1],axis=1,inplace=True)
    
        data.drop(str_col,axis=1,inplace=True)
        data=pd.concat([data,str_data],axis=1)
    return data,drop_str_cols

def test_str_col(d,cols,drop_str_cols):
    data=d.copy()
    str_cols=cols
    for str_col in str_cols:
        str_data=pd.get_dummies(data.loc[:,str_col]).astype(int)
        str_data.columns=[str_col+'_'+str(col) for col in str_data.columns]
        data.drop(str_col,axis=1,inplace=True)
        data=pd.concat([data,str_data],axis=1)

    for drop_col in drop_str_cols:
        data.drop(drop_col,axis=1,inplace=True)
        
    return data



### Outlier Treatment
def treat_outliers_zscore(data,col):
    s=data[col].describe()
    le=s['mean']-3*(s['std'])
    
    ue=s['mean']+3*(s['std'])
    d=data.copy()
    d.loc[d[col]<le,col]=le
    d.loc[d[col]>ue,col]=ue
    
    return d[col],le,ue

def treat_test_outliers_zscore(data,col,le,ue):
    d=data.copy()
    d.loc[d[col]<le,col]=le
    d.loc[d[col]>ue,col]=ue
    
    return d[col]

def make_emplength_ordinal(x):
    if isinstance(x,str):
        return 0 if ''.join(re.findall('([^ A-Za-z\+]+)',x))=='<1' else int(''.join(re.findall('([^ A-Za-z\+]+)',x)))
    return x

def narrow_transformations(data,final_columns,drop_str_cols=[]):
    df=data.copy()
    
    
    df.loc[:,'term']=df.term.apply(lambda x:int(re.findall('[^ A-Za-z]+',x)[0]))
    
    
    df.loc[:,'sub_grade']=df.sub_grade.apply(lambda x:int(re.findall('(\d)',x)[0])) #extracts 1 from A1 (A already taken care by Grade)
    
    df.loc[df.home_ownership.isin(['OTHER','NONE','ANY']),'home_ownership']='OTHER' ##broader category
    
    purpose_mapping = {
    'vacation': 'Personal',
    'debt_consolidation': 'Debt',
    'credit_card': 'Debt',
    'home_improvement': 'Home',
    'small_business': 'Business',
    'major_purchase': 'Personal',
    'other': 'Personal',
    'medical': 'Personal',
    'wedding': 'Personal',
    'car': 'Personal',
    'moving': 'Home',
    'house': 'Home',
    'educational': 'Education',
    'renewable_energy': 'Energy'
    }
    df['broad_purpose'] = df['purpose'].map(purpose_mapping)
    
    df.loc[:,'emp_length']=df.emp_length.apply(make_emplength_ordinal)
    df=convert_dates(df,'earliest_cr_line')
    df=convert_dates(df,'issue_d')
    

    final_df=pd.DataFrame(columns=final_columns,index=[0])  
    columns_left=list(df.columns.copy())
    check_cols=[]
    for ind in range(len(final_columns)):
        if ind<13:
            final_df.loc[0,final_columns[ind]]=df.loc[0,final_columns[ind]]
            columns_left.pop(columns_left.index(final_columns[ind]))
            continue
        break
    for col in columns_left: 
        check_cols.append(col+'_'+str(df.loc[0,col]))
    
    for col in check_cols:
        if col in final_columns:
            final_df.loc[0,col]=1

    final_df.replace(np.nan,0,inplace=True)
    #df.drop(['purpose'],axis=1,inplace=True) 
    
    
    return final_df,drop_str_cols


def target_encoding(df,cols,y,alpha):
    data=df.copy()
    target_maps={} #col:{'category':smoothed_value}
    reverse_target_maps={} #col:{smoothed_value:'category'}
    global_target_mean=y.mean()  
    data['y']=y
    
    for col in cols:
        tar=data.groupby([col])['y'].agg(['sum','count'])
        tar.columns=['summ','countt']
        tar['meann']=tar.summ/tar.countt
        
        tar['smoothed']=(tar.summ+alpha*global_target_mean)/(tar.countt+alpha)
        m={}
        m1={}
        for ind,row in tar.iterrows():
            m[ind]=row.smoothed
            m1[row.smoothed]=ind
            
        data.loc[:,col]=data.loc[:,col].map(m)
        target_maps[col]=m
        reverse_target_maps[col]=m1
    data.drop('y',axis=1,inplace=True)
    return data,target_maps,reverse_target_maps

def test_target_encoding(df,target_maps):
    data=df.copy()
    
    for col in target_maps:
        m=target_maps[col]
        data.loc[:,col]=data.loc[:,col].map(m)
        
    return data


def wider_transformations(data,y,is_train=True,target_maps={},num_imputer=None,cat_imputer=None,outliers={}):
    df=data.copy()
    
    #target encoding on state
    if is_train:
        alpha=10
        df,target_maps,reverse_target_maps=target_encoding(df,['state'],y,alpha)
    else:
        df=test_target_encoding(df,target_maps)
    
    #missing value treatment
    num_cols = ['mort_acc', 'pub_rec_bankruptcies','revol_util']
    cat_cols = ['emp_length']
    
    # if is_train:
    #     # Impute numerical columns
    #     num_imputer = SimpleImputer(strategy='mean')
    #     df[num_cols] = num_imputer.fit_transform(df[num_cols])

    #     # Impute categorical columns
    #     cat_imputer = SimpleImputer(strategy='most_frequent')
    #     df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
    # else:
    #     df[num_cols] = num_imputer.transform(df[num_cols])
    #     df[cat_cols] = cat_imputer.transform(df[cat_cols])
    
    
    #outlier treatment
    if is_train:
        outliers={}
        for col in df.columns:
            if (df[col].nunique()>2) or\
                ((sorted(list(df[col].unique()))!=[0,1]) and\
                (sorted(list(df[col].unique()))!=[0]) and\
                (sorted(list(df[col].unique()))!=[1])):
                
                df.loc[:,col],le,ue=treat_outliers_zscore(df,col)
                outliers[col]=[le,ue]
                
    else:
        for col in outliers:
            if col in df.columns:
                df.loc[:,col]=treat_test_outliers_zscore(df,col,outliers[col][0],outliers[col][1])
        
    if is_train:
        return df,target_maps,reverse_target_maps,num_imputer,cat_imputer,outliers
    
    
    return df

##Standardize

def filter_numeric(data,coll):
    enum_coll=list(enumerate(coll))
    cols=data.select_dtypes(include=['number']).columns
    cc=[]
    for col in cols:
        if (data[col].nunique()>2) or\
        ((sorted(list(data[col].unique()))!=[0,1]) and\
        (sorted(list(data[col].unique()))!=[0]) and\
        (sorted(list(data[col].unique()))!=[1])):
            cc.append(col)
        
    cols=cc
    numeric_cols=list(set(cols).intersection(set(coll)))
    li=[sett[1] for sett in enum_coll if sett[1] in numeric_cols]
    return li
    
def standardize_data(d,cols,strategy='standard'):
    if strategy=='standard':
        scalar=StandardScaler()   
    else:
        scalar=MinMaxScaler()
    data=d.copy()
    cols=filter_numeric(data,cols)
    data.loc[:,cols]=pd.DataFrame(scalar.fit_transform(data[cols]),columns=cols,index=data.index)
    
    return data,scalar,cols

def transform_standardize_data(d,cols,scalar):
    data=d.copy()
    
    for col in cols:
        # Get the index of the column in the scaler's fitted data columns
        col_idx = list(scalar.feature_names_in_).index(col)
        
        # Get the mean and variance for this specific column from the scaler
        col_mean = scalar.mean_[col_idx]
        col_var = scalar.var_[col_idx]
        
        # Standardize this column using its mean and variance
        data[col] = (data[col] - col_mean) / (col_var ** 0.5)
    
    return data

def revtransform_standardize_data(d,cols,scalar):
    data=d.copy()
    data.loc[:,cols]=pd.DataFrame(scalar.inverse_transform(data[cols]),columns=cols,index=data.index)
    return data