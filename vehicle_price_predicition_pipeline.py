# -*- coding: utf-8 -*-
"""Vehicle price predicition pipeline.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/15bGBs26d-mNCW33KFoKDBpjJuDkFhq8u

## Curacel ML Engineer Mini Test

**Problem**

>1. Your company is in the business of second-hand vehicle sales.
2. It helps owners and renters predict the price range to value a vehicle.
3. It helps shoppers plan for their car needs.

>Currently, there is no easy way to do #1, #2. Owners basically browse through their listings to determine what  to value their car.

**Solution**
>As a Software Engineer - ML, you are tasked to build an automated service to constantly update a prediction model with happenings in the marketplace so that, at every point in time, users can get semi-accurate prediction of the prices of proposed cars or advice them to make price adjustments for existing ones.

>Also one should also be able to predict what a car might have sold 4 months ago because the app keep tracks of the model for each time period.

>The marketplace are the available car listing services.

>Provide an authenticated api to access the service.

## Thought Process/ Assumptions:

1. Dataset related to their vehicle listing is available.
2. Use the dataset to predict price of each vehicle based on the happenings in the platform
3. Make predictions available via API calls
4. Automate the process end-to-end.
5. Persist each model after training at set intervals so that it can be used to make forward or even past predictions (using data available before the model was persisted)

## STEPS (In no order)

1. Create a carefully-generated dataset with a little noise to add a bit of randomness to it.
2. Create a simple model to make predictions
3. Persist model
4. Simulate increasing generation of newer datasets at set intervals
5. Set pipeline to re-train at set intervals and persist
6. Create inference API.

### Generate Dataset

Columns:
```
v_brand
v_model
v_year
mileage
reg_year
```
"""

import numpy as np
import pandas as pd
import datetime as dt

alp = [chr(a) for a in range(65, 89)]

def generate_data(num_brands = 20):
  v_brand = np.array(["".join(list(np.random.choice(alp, size = 3))) for i in range(num_brands)])
  # v_brand
  v_model = {}
  for brand in v_brand:
    v_model[brand] = [brand[0] + str(np.random.randint(0, 100)) for i in range(np.random.randint(50, 100))]

  # v_model
  v_year = []
  for brand in v_model.values():
    for model in brand:
      # print(model)
      v_year.append(min((2020 - ord(model[0]) + int(model[-1]) + 64), 2020))

  # v_year
  mileage = []
  for i in range(len(v_year)):
    mileage.append(np.random.randint(0, 30000) + v_year[i] * np.random.randint(20, 40))

  # mileage
  reg_year = []
  for i in range(len(mileage)):
    reg_year.append(v_year[i] + mileage[i]//30000-1)
  # reg_year

  price = []
  for i in range(len(reg_year)):
    price.append(int(max((12000 - 1000*(2020 - reg_year[i]) - mileage[i]/1000), 1200)))
  # print(price)


  brand, model = [], []
  for key in v_model:
    brand.extend([key]*len(v_model[key]))
    model.extend(v_model[key])

  data = np.array(list(zip(brand, model, mileage, v_year, reg_year, price)))
  np.random.shuffle(data)

  cols = ['v_brand', 'v_model', 'mileage', 'v_year', 'reg_year', 'price']
  return pd.DataFrame(data, columns = cols)

df = generate_data()

"""#### Create a simple model to make predictions"""

from sklearn.linear_model import  LinearRegression
from sklearn.metrics import auc, mean_squared_error
from sklearn.model_selection import train_test_split
import pickle as pk

def pre_process():
  final_data = df[['mileage', 'v_year', 'reg_year', 'price']].astype(int)
  # train, test = train_test_split(final_data, test_size = 0.20)
  train, valid = train_test_split(final_data, test_size = 0.20)

  y_train = train['price']
  x_train = train.iloc[:, :-1]

  x_valid = valid.iloc[:, :-1]
  y_valid = valid['price']

  return x_train, y_train, x_valid, y_valid


x_train, y_train, x_valid, y_valid = pre_process()

"""### Train Model and Persist After Training"""

lr = LinearRegression()
def train(lr = lr , pre_process = pre_process ):
  
  x_train, y_train, x_valid, y_valid = pre_process()
  lr.fit(x_train, y_train) # Fit model

  pred = lr.predict(x_valid) # Get predictions
  score = lr.score(x_valid, y_valid)
  pk.dump(lr, open(str(dt.date.today()) + ("_vehicle_pred.pk"),"wb"))

  return pred, score, dt.datetime.today()

"""### Visualize Result Comparing Predicted and Actual"""


"""#### Load Persisted Model For Prediction"""

# Load persisted model for inference
def predict(data, model_date):
  '''
  data variable must be a list

  date format must be 'yyy-mm-dd'
  '''
  try:
    lr = pk.load(open(str(model_date)+"_vehicle_pred.pk", 'rb'))
  except FileNotFoundError as err:
    return err

  predictions = lr.predict([data])

  return predictions

"""### Inference"""

#train(lr, pre_process )
#lr.predict(([[60000, 2006, 2008]])).item()