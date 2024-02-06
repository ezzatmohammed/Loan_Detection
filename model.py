from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer ,SimpleImputer ,KNNImputer
from sklearn.preprocessing import StandardScaler ,OneHotEncoder
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import category_encoders as ce
import pandas as pd
import numpy as np
import pickle



#read df
df=pd.read_csv('final_df.csv')



# Define the transformers separately
binary_encoder = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', ce.BinaryEncoder())
])

Onehot_encoder = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', ce.OneHotEncoder())
])

iterative_imputer = IterativeImputer(estimator=RandomForestClassifier())
# Define the column transformer
processor = ColumnTransformer(
    transformers=[
        ('IterativeImputer', iterative_imputer, ["LoanAmount"]),
        ('KNNImputer', KNNImputer(n_neighbors=3, weights='uniform'), ["Dependents", "Credit_History", "Loan_Amount_Term"]),
        ('binaryencoder', binary_encoder, ["Education", "Self_Employed", "Married"]),
        ("Onehotencoder",Onehot_encoder, ["Gender", "Property_Area"])
    ],
    remainder='passthrough'
)


X = df.drop(["Loan_ID","Loan_Status"],axis=1)
y = df["Loan_Status"]

#model
pipeline = Pipeline([
    ("TransformData", processor),
    ("scaling", StandardScaler()),
    ("model", SVC())
])


pipeline.fit(X, y)

#save model
pickle.dump(pipeline, open('model.pkl', 'wb'))
# Save the column names instead of the entire X DataFrame
with open('input_columns.pkl', 'wb') as file:
    pickle.dump(X.columns.tolist(), file)