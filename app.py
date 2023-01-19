import gradio as gr
import numpy as np
from PIL import Image
import requests
import hopsworks
import joblib
import os
import xgboost
from datetime import datetime, timedelta

project = hopsworks.login(
    api_key_value="")

mr = project.get_model_registry()
model = mr.get_model("air_model_4", version=1)
model_dir = model.download()
model = joblib.load(model_dir + "/air_model4.pkl")


def forecast():
    fs = project.get_feature_store()
    feature_view = fs.get_feature_view(
        name='miami_data_air_quality_fv',
        version=1
    )
    train_data = feature_view.get_training_data(1)[0]
    train_data = train_data.drop(labels='city_y', axis=1)
    train_data = train_data.rename(columns={'city_x': 'city'})
    train_data = train_data.sort_values(by="date", ascending=True).reset_index(drop=True)
    train_data["aqi_next_day"] = train_data.groupby('city')['aqi'].shift(1)

    X = train_data.drop(columns=["date"]).fillna(0)
    y = X.pop("aqi_next_day")
    X = X.drop(columns=['city', 'conditions']).fillna(0)

    today_data = X[1:8]
    y = model.predict(today_data)

    res = []
    for i in y:
        res.append(int(i))
    return res


# date_today = datetime.now()
# day = timedelta(days = 1)
# date_today = date_today + day
# date_today = date_today.strftime("%Y-%m-%d")
output_label = " The air quality for the next 7 days is "

demo = gr.Interface(
    fn=forecast,
    title="Air Quality Prediction",
    description="Get aqi value",
    allow_flagging="never",
    inputs=[],
    outputs=gr.Textbox(label=output_label))

demo.launch()