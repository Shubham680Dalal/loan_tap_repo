def get_square(x):
    return x*x

def test_get_square():
    assert get_square(2)==4
    assert get_square(5)==25


from loan_app_flask import app
import pytest

@pytest.fixture
def client():
    return app.test_client()

def test_hellow(client):
    resp=client.get('/')
    assert resp.status_code==200
    assert resp.data.decode()=="<p> Hello World!</p>"

def test_predict(client):
    test_data={
    "loan_amnt": 100099.0,
    "term": "60 months",
    "int_rate": 8.0,
    "installment": 464.0,
    "dti": 5.5,
    "grade": "B",
    "sub_grade": "B4",
    "emp_length": "3 years",
    "home_ownership": "OTHER",
    "purpose": "vacation",
    "initial_list_status": "w",
    "application_type": "JOINT",
    "pincode": "29597",
    "state": "AZ",
    "annual_inc": 100.0,
    "verification_status": "Not Verified",
    "issue_d": "Jan-2015",
    "earliest_cr_line": "Jan-1990",
    "open_acc": 20,
    "pub_rec": 0,
    "revol_bal": 0,
    "revol_util": 0,
    "total_acc": 0,
    "mort_acc": 0,
    "pub_rec_bankruptcies": 0
        }   
    resp=client.post("/predict",json=test_data)
    assert resp.status_code==200
    assert resp.json=={
    "Loan Status": "Fully Paid",
    "Probability": 0.9999757666410584
        }