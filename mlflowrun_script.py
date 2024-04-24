import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler as std
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
import os
import tarfile
from six.moves import urllib


DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def encode_ocean_proximity(x):
    for label in ocean_proximity_price_pivot.index.tolist():
        if x==label:
            return ocean_proximity_price_pivot.loc[x,'median_house_value']
        else:
            pass

experiment_id = mlflow.create_experiment("mlflow-housing-regression")
with mlflow.start_run(
    run_name="MAIN-RUN",
    experiment_id=experiment_id,
    tags={"version": "v1", "priority": "P1"},
    description="parent",
) as parent_run:
    
    with mlflow.start_run(
        run_name="INGESTING-DATA",
        experiment_id=experiment_id,
        description="child",
        nested=True,
    ) as child_run:
        housing = load_housing_data
        mlflow.log_artifact(housing, artifact_path="datasets")

    with mlflow.start_run(
        run_name="DATA-PREPROCESSING",
        experiment_id=experiment_id,
        description="child",
        nested=True,
    ) as child_run:
        train_data, test_data = train_test_split(housing, test_size=0.2, random_state=42)
        ocean_proximity_price_pivot = train_data[['median_house_value','ocean_proximity']].groupby('ocean_proximity').mean()
        train_data['ocean_proximity_encoded']=train_data['ocean_proximity'].apply(lambda x:encode_ocean_proximity(x))
        X=train_data.drop(['median_house_value','ocean_proximity'], axis=1)
        y=train_data['median_house_value']

        std= std()
        std.fit(X)
        train_data_model = std.transform(X)

        test_data.dropna(inplace=True)
        test_data['ocean_proximity_encoded']=test_data['ocean_proximity'].apply(lambda x:encode_ocean_proximity(x) )
        y_test = test_data['median_house_value']
        test_data.drop(['median_house_value', 'ocean_proximity'],axis=1, inplace=True)

        std.fit(test_data)
        test_data_model = std.transform(test_data)


        mlflow.log_artifact(train_data_model, artifact_path="train-data-model")
        mlflow.log_artifact(test_data_model, artifact_path="test-data-model")

    with mlflow.start_run(
        run_name="MODEL-BUILDING/SCORING",
        experiment_id=experiment_id,
        description="child",
        nested=True,
    ) as child_run:
        rfr_params = {'n_estimators': 500, 'max_depth': 8, 'max_features': 'sqrt'}
        exp = mlflow.set_experiment(experiment_name='Assign32_mlflow')

        rfr = RandomForestRegressor(**rfr_params)
        rfr.fit(train_data_model, y)
        y_pred = rfr.predict(test_data_model)
        rfr_mape = mean_absolute_percentage_error(y_test, y_pred)
    
        mlflow.log_params(rfr_params)  
    
        mlflow.log_metric("MAPE SCORE", rfr_mape)
    
        mlflow.sklearn.log_model(rfr, "ML Models")

        
        
    








        





with mlflow.start_run(experiment_id=exp.experiment_id):
    rfr = RandomForestRegressor(**rfr_params)
    rfr.fit(train_data_model, y)
    y_pred = rfr.predict(test_data_model)
    rfr_mape = mean_absolute_percentage_error(y_test, y_pred)
    
    mlflow.log_params(rfr_params)  
    
    mlflow.log_metric("MAPE SCORE", rfr_mape)
    
    mlflow.sklearn.log_model(rfr, "ML Models")