from flask import Flask, request, render_template
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
app = Flask(__name__)
forest = pickle.load(open('diabetes_model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict' , methods = ['GET', 'POST'])
def predict():

    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = forest.predict(final_features)

    output = prediction[0]

    if output == 0:
        return render_template('index.html', prediction_text= 'Diabetes : No')
    else:
        return render_template('index.html', prediction_text= 'Diabetes : Yes')


if __name__ == "__main__":
    app.run(debug=True)