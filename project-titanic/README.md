# Project Overview

### Titanic Data Exploration and Model Deployment Pipeline

This project focuses on exploring the Titanic dataset, performing data analysis and preprocessing, and using MLflow to track experiments with machine learning models. After selecting the best-performing model, we deploy it using FastAPI and Docker. The project also includes building pipelines and orchestrating tasks using Prefect to automate workflows.

## Folders Structure

- **data**: Contains the dataset used for training and testing the models.
- **models**: Contains pre-trained `.pkl` files, including the model, encoder, and `DictVectorizer`.

## **Sections Overview**

- **01-data_exploration**: Initial dataset exploration and implementation of basic machine learning models.
- **02-experiment-tracking**: Experiment tracking using MLflow, selecting the best-performing model, and saving its artifacts (model, encoder, `DictVectorizer`).
- **03-model-deployment**: Deploying the best model as a web service using FastAPI, containerizing the application with Docker, and testing the API.
- **04-pipeline-and-orchestration**: Building a pipeline with Prefect to automate tasks, visualize workflows, and schedule deployments.

## Objective

The primary objective of this project is to apply MLOps best practices by exploring the Titanic dataset, tracking model experiments, deploying a model using FastAPI, and orchestrating workflows using Prefect.

## **[MANDATORY] Pre-requisites & Setup**

> **Quick Setup Guide**:
> - Docker Desktop
> - Git
> - Conda / Miniconda / Any Python Environment Manager
> - Python 3.10
> - Install dependencies via [requirements.txt](requirements.txt)

You can find detailed setup instructions in [PREREQUISITES.md](PREREQUISITES.md).

## Timeline

**Submission Deadline**: January 6th

## **Modules and Sections**

### **[Module 1: Data Exploration](01-data-exploration)**
   - Introduced the Titanic dataset and its features.
   - Performed data cleaning and preprocessing to handle missing values.
    - Created new features like family size and title extraction.
   - Encoded categorical features using `DictVectorizer` and `LabelEncoder`.
   - Trained a RandomForestClassifier on the processed data.
   - Evaluated the model using accuracy score.

### **[Module 2: Experiment Tracking](02-experiment-tracking)**
   - Set up MLflow for tracking experiments.
   - Ran multiple experiments using MLflow to compare models.
   - Tracked parameters, metrics, and artifacts for each run.
   - Identified the best-performing model.
   - Saved the selected model, encoder, and `DictVectorizer` as artifacts.

### **[Module 3: Model Deployment](03-model-deployment)**
   - Created a FastAPI application to expose the model as an API endpoint.
   - Implemented the `/predict` endpoint to handle prediction requests.
   - Wrote a `Dockerfile` to containerize the FastAPI app.
   - Built and tested the Docker image locally.
   - Ran the Docker container and tested the `/predict` endpoint with sample data.
   - Verified the API's functionality with different test cases.

### **[Module 4: Pipelines and Orchestration](04-pipeline-and-orchestration)**
   - Set up Prefect to manage workflows and pipelines.
   - Created a `train_model_workflow` using Prefect to automate data processing, training, and evaluation.
   - Scheduled the flow to run at specific intervals.
   - Monitored the workflow's execution and reviewed logs using Prefect's UI.

## Students
- Moumen Awad (moumen.awad@edu.devinco.fr)
- Luisa Benavides (luisa.benavides@edu.devinco.fr)