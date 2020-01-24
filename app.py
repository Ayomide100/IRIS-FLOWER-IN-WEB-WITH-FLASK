from flask import Flask, render_template, url_for, request
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier


app = Flask(__name__)


@app.route('/')
def home():
	return render_template('index.html')



@app.route('/predict', methods=['POST'])
def predict():

	df = pd.read_csv(os.path.join('data', 'iris.csv'))
	c = {'Setosa':1, "Versicolor":2, "Virginica":3}
	df["species"]=df["species"].map(c)
	X = df.drop("species", axis=1)
	y = df["species"]
	data = df.to_html(classes='table table-striped table-hover', na_rep="-")

	from sklearn.model_selection import train_test_split
	X_t,X_v,y_t,y_v = train_test_split(X, y, random_state=0)

	#Naive Bayes Classifier
	clf = RandomForestClassifier(random_state=0)
	clf.fit(X_t, y_t)
	d =clf.score(X_v, y_v)
	d = np.round(d,2)


	if request.method =='POST':
		SL = request.form['SL']
		SW = request.form['SW']
		PL = request.form['PL']
		PW = request.form['PW']
		e = np.array([SL,SW,PL,PW])
		e = e[np.newaxis, :]
		my_pred = clf.predict(e)
	return render_template('results.html', prediction =my_pred, Accuracy=d, data=data)




if __name__ == '__main__':
	app.run(debug=True)
