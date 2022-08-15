#from src.utils.all_utils import read_yaml, create_directory, save_local_df
from src.utils.common import read_yaml, create_directories, save_json
import argparse
import pandas as pd
import logging
import os
from sklearn.linear_model import ElasticNet
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

STAGE = "Evaluate" ## <<< change stage name

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def evaluate_metrics(actual_values, predicted_values):
    rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
    mae = mean_absolute_error(actual_values, predicted_values)
    r2 = r2_score(actual_values, predicted_values)
    return rmse, mae, r2

def main(config_path, params_path):
    ## read config files
    config = read_yaml(config_path)
    params = read_yaml(params_path)

    artifacts = config["artifacts"]
    artifacts_dir = artifacts['ARTIFACTS_DIR']
    split_data_dir = artifacts['SPLIT_DATA_DIR']

    split_data_dir_path = os.path.join(artifacts_dir, split_data_dir)
    test_data_path = os.path.join(split_data_dir_path, artifacts["TEST"])

    #train_data_filename = config["artifacts"]["train"]
    #train_data_path = os.path.join(artifacts_dir, split_data_dir, train_data_filename)
    #train_data = pd.read_csv(train_data_path)
    #train_y = train_data["quality"]
    #train_x = train_data.drop("quality", axis=1)

    #alpha = params["model_params"]["ElasticNet"]["alpha"]
    #l1_ratio = params["model_params"]["ElasticNet"]["l1_ratio"]
    #andom_state = params["base"]["random_state"]

    test_df = pd.read_csv(test_data_path)

    target_col = "quality"
    test_y = test_df[target_col]
    test_x = test_df.drop(target_col, axis=1)

    model_dir = artifacts["MODEL_DIR"]
    model_name = artifacts["MODEL_NAME"]
    model_dir_path = os.path.join(artifacts_dir, model_dir)
    model_file_path = os.path.join(model_dir_path, model_name)

    lr = joblib.load(model_file_path)

    predicted_values = lr.predict(test_x)
    rmse,mae,r2 = evaluate_metrics(
        actual_values= test_y,
        predicted_values = predicted_values
    )

    scores = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }

    scores_file_path = config["scores"]
    save_json(scores_file_path, scores)

if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="configs/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")

    parsed_args = args.parse_args()

    main(config_path=parsed_args.config, params_path=parsed_args.params)