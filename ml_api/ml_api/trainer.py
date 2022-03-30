import pandas as pd
import numpy as np
import joblib


# Preprocessing

from sklearn.preprocessing import RobustScaler,  OneHotEncoder, OrdinalEncoder,  MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_selector
from imblearn.over_sampling import SMOTE


# Model
from imblearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestClassifier
from ml_api.data import storage_upload, drop_features, split_data, get_data_from_gcp



class Trainer(object):
    def __init__(self, X, y):
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        # Defining columns to Imput and Scale
        impute_col = ['avg_payment_span_0_3m','num_active_div_by_paid_inv_0_12m','num_arch_written_off_12_24m', \
                'num_arch_written_off_0_12m','account_days_in_dc_12_24m','account_days_in_rem_12_24m','account_days_in_term_12_24m', \
               'sum_capital_paid_account_12_24m','sum_capital_paid_account_0_12m','recovery_debt']
        scale_col = ['sum_paid_inv_0_12m', 'time_hours', 'max_paid_inv_0_12m' ]

        # Imputing, scaling and Encoding data. Dropping the remaining columns ('merchant_category'  and 'name_in_email')


        preproc = make_column_transformer(
            (OneHotEncoder(handle_unknown='ignore'),
                ['merchant_group']
            ),
            (OrdinalEncoder(),
                ['has_paid']
             ),
            (SimpleImputer(strategy="median"),
                impute_col
             ),
            (RobustScaler(),
                scale_col
             ),
            (MinMaxScaler(),
                make_column_selector(dtype_include=['int64'])
                ),
                remainder='drop'
            )


        self.pipeline = make_pipeline (preproc ,SMOTE(), RandomForestClassifier(n_estimators=100))
        self.pipeline.fit(self.X, self.y)



    def run(self):
        self.set_pipeline()
        self.pipeline.fit(self.X, self.y)

        return self.pipeline

    def predict(self, X_pred):
        """Trains the model with RandomForestClassifier using the preprocessing pipeline.

        """
        self.set_pipeline()
        y_pred = self.pipeline.predict(X_pred)

        return y_pred

    def save_model(self, model):
        """Saves the model and the uploads it to the cloud.

        Args:
            model (joblib): Trained model.
        """
        joblib.dump(model, 'model.joblib')
        print("model.joblib saved locally")
        storage_upload(rm=False)



if __name__ == "__main__":
    #df = get_local_data() # gets data locally


    df = get_data_from_gcp() # gets data from the cloud
    df = drop_features(df) # drop column highly correlated and with null values
    train_df,predict_df = split_data(df) #split data frame in two : one to train the model, the other to make the prediction

    trainer = Trainer(train_df.drop('default', axis = 1), train_df['default'])
    model = trainer.run()
    y_pred = trainer.predict(predict_df.drop('default', axis = 1))
    #print(y_pred)
    trainer.save_model(model)
