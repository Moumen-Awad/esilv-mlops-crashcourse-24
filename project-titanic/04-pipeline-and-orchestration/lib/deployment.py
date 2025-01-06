import os
# Retrieve paths of dataset and models
from config import DATA_DIRPATH, MODELS_DIRPATH
from prefect import serve
from workflows import batch_predict_workflow, train_model_workflow

if __name__ == "__main__":
    train_model_deployment = train_model_workflow.to_deployment(
        name="Model training Deployment",
        version="0.1.0",
        tags=["training", "model"],
        cron="0 0 * * 0",
        parameters={
            "data_filepath": os.path.join(DATA_DIRPATH, "train_and_test2.csv"),
            "artifacts_filepath": MODELS_DIRPATH,
        },
    )

    batch_predict_deployment = batch_predict_workflow.to_deployment(
        name="Batch predict Deployment",
        version="0.1.0",
        tags=["inference"],
        interval=600,
        parameters={
            "input_filepath": os.path.join(DATA_DIRPATH, "train_and_test2.csv"),
            "artifacts_filepath": MODELS_DIRPATH,
        },
    )
    serve(train_model_deployment, batch_predict_deployment)
