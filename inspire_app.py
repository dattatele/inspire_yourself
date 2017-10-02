#import numpy as np
#from flask import Flask, abort, jsonify, request, render_template
from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import pickle
import sqlite3
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

#naive_model = pickle.load(open("C://Users//datta//inspire_app//model.pkl","rb"))
# Preparing the classifier
cur_dir = os.path.dirname(__file__)
clf = pickle.load(open(os.path.join(cur_dir,
                        'inspireclassifier/pkl_objects/classifier.pkl'), 'rb'))
db = os.path.join(cur_dir, 'inspire.sqlite')

def classify(document):
    df = pd.read_csv("songform.csv", encoding="cp1252")
    corpus = df['lyrics']
    vectorizer = TfidfVectorizer()
    vec = vectorizer.fit_transform(corpus).toarray()
    label = {0: 'Inspiring', 1: 'depressing'}
    X = vectorizer.transform([document])
    y = clf.predict(X)[0]
    proba = np.max(clf.predict_proba(X))
    return label[y], proba

def train(document, y):
    X = vectorizer.transform([document])
    clf.partial_fit(X, [y])

def sqlite_entry(path, document, y):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("INSERT INTO inspire_db (lyrics, category, date)"\
                    " VALUES (?, ?, DATETIME('now'))", (document, y))
    conn.commit()
    conn.close()

app = Flask(__name__)

class ReviewForm(Form):
    songtype = TextAreaField('',
                 [validators.DataRequired(), validators.length(min=15)])

@app.route('/')
#@app.route('/inspire', methods=['POST'])
def index():
    form = ReviewForm(request.form)
    return render_template('reviewform.html', form=form)

@app.route('/inspire', methods=['POST'])
def results():
    form = ReviewForm(request.form)
    if request.method == 'POST' and form.validate():
        review = request.form['songtype']
        y, proba = classify(review)
        return render_template('inspire.html',
    content=review,
    prediction=y,
    probability=round(proba*100, 2))
    return render_template('reviewform.html', form=form)

@app.route('/thanks', methods=['POST'])
def feedback():
    feedback = request.form['feedback_button']
    review = request.form['review']
    prediction = request.form['prediction']
    inv_label = {'Inspiring': 0, 'depressing': 1}
    y = inv_label[prediction]
    if feedback == 'Inspiring':
        y = int(not(y))
    train(review, y)
    sqlite_entry(db, review, y)
    return render_template('thanks.html')

#def predict():
     # Error checking
     #data = request.get_json(force=True)

     # Convert JSON to numpy array
     ##predict_request = np.array(predict_request)

     # Predict using the random forest model
     #y = naive_model.predict(predict_request)

     # Return prediction
     #output = [y[0]]
     #return jsonify(results=output)

if __name__ == '__main__':
    app.run(debug=True)
     #app.run(port = 9000, debug = True)
