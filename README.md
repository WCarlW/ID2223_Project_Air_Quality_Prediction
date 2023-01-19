# ID2223_Project_Air_Quality_Prediction

The goal of this project is to predict Miami air quality index (AQI) in the next 7 days with historical air quality index and weather data. In this project we collect historical weather and air quality data from AirNow, EPA (US Environmental Protection Agency), and VisualCrossing website. We use XGBoost Regressor to train a model using both historical weather data and air quality measurements to predict future air quality. Then, we build a Hugging Face space where you can see the predictions of air quality for that location for the next 7 days.

## Hugging Face Space

App url:
https://huggingface.co/spaces/howlbz/air3

## Data:

### Historical AQI Data

The historical aqi data is from the EPA website: 

https://aqs.epa.gov/aqsweb/documents/data_api.html#format

### Current AQI Data

The current aqi data is from the AirNow website: 

https://www.airnow.gov/

### Weather Data

The weather data is from the VisualCrossing website:

https://www.visualcrossing.com/

