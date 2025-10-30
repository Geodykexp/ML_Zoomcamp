import numpy as np
import pandas as pd
import sklearn as sk

import pickle


model_file = 'pipeline_v1.bin'


with open(model_file, 'rb') as f_in:
    dv,model = pickle.load(f_in)


record = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}


X = dv.transform([record])
y_pred = model.predict_proba(X)[0,1]

print('input', record)
print('probability', y_pred)





