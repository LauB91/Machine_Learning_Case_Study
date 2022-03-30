import pandas as pd
import numpy as np
import joblib

from ml_api.data import storage_upload, get_local_data, drop_features, split_data, get_data_from_gcp

# Preprocessing
from sklearn.preprocessing import RobustScaler, OneHotEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_selector

# Model
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import make_pipeline


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


        self.pipeline = make_pipeline (preproc, RandomForestClassifier(n_estimators=100))
        #self.pipeline.fit(self.X, self.y)



    def run(self):
        self.set_pipeline()
        self.pipeline.fit(self.X, self.y)

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

    def save_model_locally(self):
        """Save the model into a .joblib format"""
        joblib.dump(self.pipeline, 'model.joblib')
        print(("model.joblib saved locally"))


if __name__ == "__main__":
    #df = get_local_data() # gets data locally


    df = get_data_from_gcp() # gets data from the cloud
    df = drop_features(df) # drop column highly correlated and with null values
    train_df,predict_df = split_data(df) #split data frame in two : one to train the model, the other to make the prediction

    trainer = Trainer(train_df.drop('default', axis = 1), train_df['default'])
    trainer.run()
    model = trainer.save_model_locally()
    print (model)

   # y_pred = trainer.predict(predict_df)


    trainer.save_model(model)
