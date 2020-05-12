import vehicle_price_predicition_pipeline as retrain
from flask import Flask, make_response, request, redirect, url_for, flash, jsonify, render_template
import numpy as np
import pickle as pk
import json
import datetime as dt
from functools import wraps
from flask_apscheduler import APScheduler


app = Flask(__name__)
date = dt.date.today()-dt.timedelta(1)
score = 0

def auth_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if auth and auth.username == 'admin' and auth.password == 'vehicle2020':
            return f(*args, **kwargs)
        return make_response("Could not verify!", 401, {'WWW-Authenticate': 'Basic realm = "Login Required"'})
    return decorated

@app.route('/')
@auth_required
def index():
    return render_template('Home.html', date = f"Model last trained on {date}", score = f"with an accuracy of {score:.2%}")
    
@app.route('/home')
def home():
    return render_template('Home.html', date = f"Model last trained on {date}", score = f"with an accuracy of {score:.2%}")


@app.route('/api/', methods=['POST'])
# Load persisted model for inference
@auth_required
def predict():
    '''
    data variable must be a list

    date format must be 'yyy-mm-dd'
    '''
    
    response = request.form
    model_date = response.get("date")
    data = list(response.values())
    data.remove(model_date)
    try:
        
        data = list(map(int, data))
    except ValueError:
        return render_template('Home.html', error = "Oops! Enter Valid data types", date = f"Model last trained on {date}", score = f"with an accuracy of {score:.2%}")

    if request.method == 'POST':
      #write your function that loads the model
    
        if model_date:
            model_date = dt.date.today().strftime("%Y-%m-%d")
            try:
                model = pk.load(open(str(model_date)+"_vehicle_pred.pk", 'rb'))
            except FileNotFoundError as err:
                return render_template('Home.html', error = "Oops! No model has been trained for that period", date = f"Model last trained on {date}", score = f"with an accuracy of {score:.2%}")

        try:
            predictions = model.predict([data])
        except:
            return render_template('Home.html', error = "Model could not make a prediction")

    return render_template("predictor_page.html", prediction = f"$ {predictions.item():.0f}", date = f"Model last trained on{date}", score = f"with an accuracy of {score:.2%}")

@app.route('/train', methods = ["POST"])
def train():
    global date, score
    _, score, date = retrain.train()
    return render_template('Home.html', error = "Model could not make a prediction", date = f"Model was last trained on {date}", score = f"with an accuracy of {score:.2%}")
        
if __name__ == '__main__':
    scheduler = APScheduler()
    scheduler.add_job(func=train, args=['job run'], trigger='interval', id='job', hours=1)
    scheduler.start()
    model_date = dt.date.today().strftime("%Y-%m-%d")
    model = pk.load(open(str(model_date)+"_vehicle_pred.pk", 'rb'))
    
    app.run(debug=True)
