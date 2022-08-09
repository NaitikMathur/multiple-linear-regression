import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


app = Flask(__name__)
model = pickle.load(open('Summer internship 2022 Project 2.pkl','rb')) 
dataset = pd.read_csv('house-prices.csv')

from sklearn.impute import SimpleImputer 
imputer = SimpleImputer(missing_values= np.NAN, strategy= 'mean', fill_value=None, verbose=1, copy=True)   
imputer = imputer.fit(dataset.iloc[:,1:3])  
dataset.iloc[:,1:3]= imputer.transform(dataset.iloc[:,1:3])

X = dataset.iloc[:, [0,2,3,4,5,6,7]].values 
y = dataset.iloc[:, 1].values

from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 5] = labelencoder_X.fit_transform(X[:, 5])

from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 6] = labelencoder_X.fit_transform(X[:, 6])

#dummy encoding.
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [6])],remainder='passthrough')
X=columnTransformer.fit_transform(X)
 
# Dummy Variable trapping
X = X[:, 1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

@app.route('/')
def home():
  
    return render_template("index.html")
  
@app.route('/predict',methods=['GET'])
def predict():
    
    
    '''
    For rendering results on HTML GUI
    '''
    home = float(request.args.get('home'))
    SqFt = float(request.args.get('SqFt'))
    bed = float(request.args.get('bed'))
    bath = float(request.args.get('bath'))
    offer = float(request.args.get('offers'))
    neighborhood = float(request.args.get('neighborhood'))
    

    y_pred=model.predict(sc_X.transform(np.array([[0, 1, home, SqFt, bed, bath, offer, neighborhood]])))
    
        
    return render_template('index.html', prediction_text='Regression Model  has predicted house price : {}'.format(y_pred))


if __name__ == "__main__":
    app.run(debug=True)