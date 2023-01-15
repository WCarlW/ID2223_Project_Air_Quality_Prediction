# ID2223_Project_Air_Quality_Prediction

The goal of this project is to predict Miami air quality. In this project we collect historical weather and air quality data from AirNow and EPA (US Environmental Protection Agency) website. We train a model with acceptable performance (XGBoost as a baseline) that uses both historical weather data and air quality measurements to predict future air quality (using future weather predictions) we also build a Hugging Face space where you can see the predictions of air quality for that location for the next 7 days.

## Hugging Face Space

https://huggingface.co/spaces/howlbz/air3

### Data-centric approach:

The following webpage contains dataset which we used:

https://aqs.epa.gov/aqsweb/documents/data_api.html
https://www.airnow.gov/
