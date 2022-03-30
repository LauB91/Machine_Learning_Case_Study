import pandas as pd
import os
import joblib
from google.cloud import storage

LOCAL_PATH='raw_data/dataset.csv'
GCP_PATH = 'data/dataset.csv'
BUCKET_NAME='ml_api_lau'

# model folder name (will contain the folders for all trained model versions)
MODEL_NAME = 'ml_training'

# model version folder name (where the trained model.joblib file will be stored)
MODEL_VERSION = 'v1'


def get_local_data():

    """Fucntion to get the data locally

    Returns:
        pandas.DataFrame:
    """

    return pd.read_csv(LOCAL_PATH,sep = ';').set_index('uuid')

def get_data_from_gcp():
    """Function to get the data from the cloud

    Returns:
        pandas.DataFrame
    """
    path = f"gs://{BUCKET_NAME}/{GCP_PATH}"
    return pd.read_csv(path,sep = ';').set_index('uuid')

def drop_features(df):
    """Function to delete the columns with too many null values and multicolinearity

    Returns:
        pandas.Dataframe
    """
    null_list = ['account_incoming_debt_vs_paid_0_24m','account_status','account_worst_status_0_3m','account_worst_status_12_24m',\
                 'account_worst_status_3_6m','account_worst_status_6_12m','worst_status_active_inv']

    corr_list = ['max_paid_inv_0_24m', 'num_arch_ok_12_24m', 'status_max_archived_0_12_months', 'status_last_archived_0_24m', \
                'avg_payment_span_0_12m', 'status_max_archived_0_12_months','account_amount_added_12_24m',  \
                              'status_max_archived_0_24_months']


    return df.drop(null_list + corr_list , 1 )

def split_data(df):
    """Function to split dataframe between training data and predict data

    Returns:
        two pandas.Dataframe : train_df, predict_df
    """

    return df[df['default'].isnull() == False] , df[df['default'].isnull() == True]



def storage_upload(rm=True):
    """Function to save model & preprocessing outcome to the cloud

    Args:
        rm (bool, optional): Especifies if the files should be removed from local folder. Defaults to True.
    """
    client = storage.Client().bucket(BUCKET_NAME)

    local_model_name = 'model.joblib'
    model_storage_location = f"models/{MODEL_NAME}/{MODEL_VERSION}/{local_model_name}"
    blob = client.blob(model_storage_location)
    blob.upload_from_filename('model.joblib')
    print(f"=> model.joblib uploaded to bucket {BUCKET_NAME} inside {model_storage_location}")


    if rm:
        os.remove('model.joblib')


def get_model_from_gcp():
    """Function to get the trained model from the cloud

    Returns:
        joblib: Trained model
    """
    client = storage.Client().bucket(BUCKET_NAME)

    local_model_name = 'model.joblib'
    model_storage_location = f"models/{MODEL_NAME}/{MODEL_VERSION}/{local_model_name}"
    blob = client.blob(model_storage_location)
    blob.download_to_filename('model.joblib')
    return joblib.load('model.joblib')
