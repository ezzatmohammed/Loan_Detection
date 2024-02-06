import streamlit as st
import pandas as pd
import pickle

with open("model.pkl", 'rb') as model_file:
    model = pickle.load(model_file)

with open('input_columns.pkl', 'rb') as file:
    input_columns = pickle.load(file)



def predictions(Gender, Married, Dependents, Education , Self_Employed , ApplicantIncome, CoapplicantIncome, LoanAmount,Loan_Amount_Term,Credit_History,Property_Area):

    df = pd.DataFrame(columns=input_columns)
    df.at[0,'Gender'] = Gender
    df.at[0,'Married'] = Married
    df.at[0,'Dependents'] = Dependents
    df.at[0,' Education'] =  Education
    df.at[0,'Self_Employed'] = Self_Employed
    df.at[0,'ApplicantIncome'] = ApplicantIncome
    df.at[0,'CoapplicantIncome'] = CoapplicantIncome
    df.at[0,'LoanAmount'] = LoanAmount
    df.at[0,'Loan_Amount_Term'] = Loan_Amount_Term
    df.at[0,'Credit_History'] = Credit_History
    df.at[0,'Property_Area'] = Property_Area

    result = model.predict(df)
    return result


def main():

    
    Gender_option = ["Male","Female"]
    Gender = st.selectbox ("select a Gender",Gender_option)

    Married_option = ["Yes","No"]
    Married = st.selectbox ("select a Married", Married_option)

    Dependents_option = [0,1,2,3]
    Dependents = st.selectbox ("select a Dependents", Dependents_option)

    Education_option = ["Graduate","Not Graduate"]
    Education = st.selectbox ("select a Education", Education_option)

    Self_Employed_option = ["Yes","No"]
    Self_Employed = st.selectbox ("select a Self Employed", Self_Employed_option)

    ApplicantIncome = st.number_input("Applicant Income")

    CoapplicantIncome = st.number_input("Coapplicant Income")

    LoanAmount = st.number_input("Loan Amount")


    Loan_Amount_Term_option = [360,180,480,300]
    Loan_Amount_Term = st.selectbox("select a Loan Amount Term", Loan_Amount_Term_option)

    Credit_History_option = [1,0]
    Credit_History = st.selectbox("select a Credit History 1 if good, 0 if problem", Credit_History_option)

    Property_Area_option = ["Urban","Rural","Semiurban"]
    Property_Area = st.selectbox("select a Property Area",Property_Area_option)


    if st.button("Predict"):
        result = predictions(Gender, Married, Dependents, Education , Self_Employed , ApplicantIncome, CoapplicantIncome, LoanAmount,Loan_Amount_Term,Credit_History,Property_Area)
        result_label = "Approved" if result == 1 else "Rejected"
        st.write(f"The prediction result is: {result_label}")




main()






