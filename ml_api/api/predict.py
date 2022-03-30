from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ml_api.data import get_data_from_gcp, get_model_from_gcp, drop_features, split_data
import pandas as pd


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    """Home page for API

    Returns:
        dic: Dictonary just saying 'hello word'
    """
    return {"greeting": "this is not a landing page"}


@app.get("/predict")
def predict(uuid_list):
    """Recommendation part of the API

    Args:
        game (str): predict_df

    Returns:
        dic: Dictonary with a list of default value for each uuid
    """
    uuid_list = list(uuid_list.split(','))
    df = get_data_from_gcp()
    df = drop_features(df)
    train_df,predict_df = split_data(df)

    model = get_model_from_gcp()
    results = {'prediction' : [{'uuid' : id, 'pd' : model.predict_proba(predict_df.drop('default',1).loc[[id]])[0][0]} \
        for id in uuid_list if id in predict_df.index  ]}

    return results
