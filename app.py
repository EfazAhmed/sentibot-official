from urllib import request
from flask import Flask, render_template, request
from create_model import create_feature, vectorizer
import pickle
import random


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
    all_probas["joy"] = predicted_probas[4]
    all_probas["fear"] = predicted_probas[2]
    all_probas["anger"] = predicted_probas[0]
    all_probas["sadness"] = predicted_probas[5]
    all_probas["disgust"] = predicted_probas[1]
    all_probas["shame"] = predicted_probas[6]
    all_probas["guilt"] = predicted_probas[3]

    joy_proba = round(float(all_probas["joy"]), 2) * 100
    fear_proba = round(float(all_probas["fear"]), 2)  * 100
    anger_proba = round(float(all_probas["anger"]), 2)  * 100
    sadness_proba = round(float(all_probas["sadness"]), 2)  * 100
    disgust_proba = round(float(all_probas["disgust"]), 2)  * 100
    shame_proba = round(float(all_probas["shame"]), 2)  * 100
    guilt_proba = round(float(all_probas["guilt"]), 2)  * 100

    rel_probas = [joy_proba, fear_proba, anger_proba, sadness_proba, disgust_proba, shame_proba, guilt_proba]
    probas = [joy_proba, fear_proba, anger_proba, sadness_proba, disgust_proba, shame_proba, guilt_proba]

    for i in range(len(probas)):
        temp = (probas[i]/(100 - probas[i]))*100
        probas[i] = round(temp, 2)

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

    quotes = {"joy": [
            "The purpose of our lives is to be happy.",
            "No medicine cures what happiness cannot.",
            "Happiness is when what you think, what you say, and what you do are in harmony.",

        ],
        "fear": [
            "Being scared is part of being alive. Accept it. WAlk through it.",
            "You can't stop being afraid just by pretending everything that scares you isn't there",
            "fear kills more dreams than failure ever will."
        ],
        "anger": [
            "where there is anger, there is always pain underneath.",
            "anger never solves anything.",
            "angry people are not always wise."
        ],
        "sadness": [
            "People cry, not because they're weak. It's because they've been strong for too long.",
            "sadness is also a kind of defense.",
            "be strong because things will get better. It may be stormy now, but it never rains forever."
        ],
        "disgust": [
            "disgust is the appropriate response to most situations.",
            "disgust is so reassuring; it feels like a moral proof.",
            "the greatest pleasures are only narrowly separated from disgust."
        ],
        "shame": [
            "shame corrodes the very part of us that believes we are capable of change.",
            "shame should be reserved for the things we choose to do, not the circumstances that life puts on us.",
            "shame cannot survive being spoken. It cannot survive empathy."
        ],
        "guilt": [
            "mistakes are always forgivable, if one has the courage to admit them.",
            "guilt is perhaps the most painful companion of death.",
            "guilt can either hold you back from growing or it can show you what you need to shift in your life."
        ]
    }


    quote = "\"" + random.choice(quotes[prediction]) + "\""
    predict = outputs[prediction][0]
    emotions = ["Happy", 'Scared', "Anger", "Sad", "Disgust", "Shame", "Guilt"]

    colors = ['rgb(255, 221, 0)',
              'rgb(125, 63, 152)',
              'rgb(190, 0, 39)',
              'rgb(0, 121, 193)',
              'rgb(132, 189, 0)',
              'rgb(255, 153, 51)',
              'rgb(200,200,200)']

    images = ['images/robot_yellow.png',
              'images/robot_purple.png',
              'images/robot_red.png',
              'images/robot_blue.png',
              'images/robot_green.png',
              'images/robot_orange.png',
              'images/robot_grey.png']

    if prediction == "joy":
        color = colors[0]
        image = images[0]
    elif prediction == "fear":
        color = colors[1]
        image = images[1]
    elif prediction == "anger":
        color = colors[2]
        image = images[2]
    elif prediction == "sadness":
        color = colors[3]
        image = images[3]
    elif prediction == "disgust":
        color = colors[4]
        image = images[4]
    elif prediction == "shame":
        color = colors[5]
        image = images[5]
    elif prediction == "guilt":
        color = colors[6]
        image = images[6]
    else:
        image = 'images/robot_white.png'
        color = "white"

    print(rel_probas)
    print(emotions)

    return render_template('analysis.html', labels=emotions, user=text, rel_values=rel_probas, values=probas, color=color, text=quote, prediction=predict, image=image)

if __name__ == '__main__':
    app.run(port=5000, debug=True)