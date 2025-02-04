import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import gradio
from fastapi import FastAPI, Request, Response

import random
import numpy as np
import pandas as pd
from bikeshare_model.processing.data_manager import load_dataset, load_pipeline
from bikeshare_model import __version__ as _version
from bikeshare_model.config.core import config
from sklearn.model_selection import train_test_split
from bikeshare_model.predict import make_prediction


# FastAPI object
app = FastAPI()


################################# Prometheus related code START ######################################################
import prometheus_client as prom

import pandas as pd
from sklearn.metrics import r2_score

# Metric object of type gauge
r2_metric = prom.Gauge('bikeshare_r2_score', 'R2 score for random 100 test samples')


# LOAD TEST DATA
data = load_dataset(file_name = config.app_config_.training_data_file)
# divide train and test
train_data, test_data = train_test_split(
    data,
    test_size = config.model_config_.test_size,
    random_state=config.model_config_.random_state,
)


# Function for updating metrics
def update_metrics():
    test = test_data.sample(100)   # sample 100 rows from test data randomly
    test_features = test[config.model_config_.features]       # features
    test_cnt = test[config.model_config_.target].values       # target
    test_pred = make_prediction(input_data=test_features)['predictions']    # make prediction
    
    r2 = round(r2_score(test_cnt, test_pred), 3)      # calculate r2_score
    
    r2_metric.set(r2)      # update prometheus metric object


@app.get("/metrics")
async def get_metrics():
    update_metrics()
    return Response(media_type="text/plain", content= prom.generate_latest())

################################# Prometheus related code END ######################################################


# UI - Input components
in_dteday = gradio.Textbox(lines=1, placeholder=None, value="2012-11-6", label='Enter date (YYYY-MM-DD)')
in_season = gradio.Radio(list(config.model_config_.season_mappings.keys()), type="value", label='Season')
in_hr = gradio.Dropdown(list(config.model_config_.hr_mappings.keys()), label="Hour")
in_holiday = gradio.Radio(['Yes', 'No'], type="value", label="Whether the day is considered a holiday")
in_weekday = gradio.Dropdown(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], label="Day of the week")
in_workingday = gradio.Radio(['Yes', 'No'], type="value", label="Whether the day is neither a weekend nor holiday")
in_weathersit = gradio.Radio(list(config.model_config_.weathersit_mappings.keys()), type="value", label="Weather situation")
in_temp = gradio.Textbox(lines=1, placeholder=None, value="16", label='Temperature in Celsius')
in_atemp = gradio.Textbox(lines=1, placeholder=None, value="17.5", label='"Feels like" Temperature in Celsius')
in_hum = gradio.Textbox(lines=1, placeholder=None, value="30", label='Relative humidity')
in_windspeed = gradio.Textbox(lines=1, placeholder=None, value="10", label='Wind speed')


# UI - Output component
out_label = gradio.Textbox(type="text", label='Prediction Count', elem_id="out_textbox")

# Label prediction function
def get_output_label(in_dteday, in_season, in_hr, in_holiday, in_weekday, in_workingday, in_weathersit, in_temp, in_atemp, in_hum, in_windspeed):
    
    input_df = pd.DataFrame({'dteday': [in_dteday], 
                             'season': [in_season], 
                             'hr': [in_hr], 
                             'holiday': [in_holiday], 
                             'weekday': [in_weekday],
                             'workingday': [in_workingday], 
                             'weathersit': [in_weathersit], 
                             'temp': [float(in_temp)], 
                             'atemp': [float(in_atemp)], 
                             'hum': [float(in_hum)], 
                             'windspeed': [float(in_windspeed)]})
    
    result = make_prediction(input_data=input_df.replace({np.nan: None}))["predictions"]
    pred_count = int(result[0])
    return pred_count


# Create Gradio interface object
iface = gradio.Interface(fn = get_output_label,
                         inputs = [in_dteday, in_season, in_hr, in_holiday, in_weekday, in_workingday, in_weathersit, in_temp, in_atemp, in_hum, in_windspeed],
                         outputs = [out_label],
                         title="Bike Rental Count Prediction API",
                         description="Predictive model that predicts the bike rental count based on the environmental and seasonal settings",
                         flagging_mode='never'
                         )

# Mount gradio interface object on FastAPI app at endpoint = '/'
app = gradio.mount_gradio_app(app, iface, path="/")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 
