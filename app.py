from urllib import request
from flask import Flask, render_template, request
from create_model import create_feature, vectorizer
import pickle
import pandas as pd
import json
import plotly
import plotly.express as px


app = Flask(__name__, static_url_path="", static_folder="static", template_folder="templates")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/analysis', methods=['GET', 'POST'])
def analysis():

    text = request.form["analysis"]
    print(text)
    path_to_model = "models/rforest.pkl"
    model = pickle.load(open(path_to_model, "rb"))

    features = create_feature(text, nrange=(1,4))
    features = vectorizer.transform(features)

    prediction = str(model.predict(features)[0])
    predicted_probas = model.predict_proba(features)
    predicted_probas = predicted_probas.tolist()[0]

    all_probas = {}
    all_probas["Happiness"] = predicted_probas[4]
    all_probas["Fear"] = predicted_probas[2]
    all_probas["Anger"] = predicted_probas[0]
    all_probas["Sadness"] = predicted_probas[5]
    all_probas["Disgust"] = predicted_probas[1]
    all_probas["Shameful"] = predicted_probas[6]
    all_probas["Guilt"] = predicted_probas[3]

    print("sum", sum(list(all_probas.values())))
    joy_proba = round(float(all_probas["Happiness"]), 2) * 100
    fear_proba = round(float(all_probas["Fear"]), 2)  * 100
    anger_proba = round(float(all_probas["Anger"]), 2)  * 100
    sadness_proba = round(float(all_probas["Sadness"]), 2)  * 100
    disgust_proba = round(float(all_probas["Disgust"]), 2)  * 100
    shame_proba = round(float(all_probas["Shameful"]), 2)  * 100
    guilt_proba = round(float(all_probas["Guilt"]), 2)  * 100

    emotions = ["Happiness", "Fear", "Anger", "Sadness", "Disgust", "Shameful", "Guilt"]
    values = [joy_proba, fear_proba, anger_proba, sadness_proba, disgust_proba, shame_proba, guilt_proba]

    df = pd.DataFrame({'Emotion': emotions, 'Percentage': values})

    fig = px.bar(df, x='Emotion', y='Percentage')
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    outputs = {"joy": [
            "Happiness"
        ],
        "fear": [
            "Fear"
        ],
        "anger": [
            "Anger"
        ],
        "sadness": [
            "Sadness"
        ],
        "disgust": [
            "Disgust"
        ],
        "shame": [
            "Shameful"
        ],
        "guilt": [
            "Guilt"
        ]
    }

    predict = outputs[prediction][0]

    

    return render_template('analysis.html', user=text, prediction=predict, graphJSON=graphJSON)

if __name__ == '__main__':
    app.run(port=5000, debug=True)