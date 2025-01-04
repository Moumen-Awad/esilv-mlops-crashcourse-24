# Overview

## Objective

Titanic Data Exploration and Model Experimentation

## [MANDATORY] Pre-requisites & setup

> [!Note]
> TL; DR
> - Docker Desktop
> - Git
> - Conda / Minconda / another Python environment manager
> - Python 3.10
> - Install [./requirements.txt](requirements.txt)

You can find all pre-requisites and setup instructions [here](PREREQUISITES.md).

## Timeline

Course start: October 9th 
Course end: January 6th

## Syllabus

## [Module 1: Introduction to MLOps](lessons/00-intro)

* What is MLOps
* Course overview
* Coding best practices
* Prerequisites and setup
* Running example: NY Taxi trips dataset


## [Module 2: Experiment Tracking](lessons/01-model-and-experiment-management)

* Experiment tracking intro
* What is MLflow
* Experiment tracking with MLflow
* Saving and loading models with MLflow
* Model registry
* Practice


## [Module 3: Model Deployment](lessons/02-model-deployment)

* Web service: model deployment with FastAPI
* Docker: containerizing a web service
* Practice


## [Module 4: Pipelines and Orchestration](lessons/03-pipeline-and-orchestration)

* Tasks, Flows, Deployments
* From notebooks to Workflows
* Workflows orchestration with prefect
* Practice


## [Module 5: Life Cycle Management]

* Model monitoring
* Model retraining
* Concept drift
* Data drift & data management


## [Module 6: Project]

* End-to-end project with all the things above

### Titanic Data Exploration and Model Experimentation

This project explores the Titanic dataset, performs data analysis and preprocessing, and uses MLflow to track experiments with different machine learning models.

* data folder: contains the dataset used for training and testing of the model
* mlflow folder: cointains the different models used and its artefacts

Sections: 
* 00-data_exploration: intial exploration of the data 
* 01-experiment-tracking: implementation of the MLflow runs 

## Instructors

- ROHART Capucine (capucine.rohart@artefact.com)
