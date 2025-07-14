⚠️ Hinweis zur Datenverfügbarkeit

Dieses Repository basiert auf einem Projekt, das im Rahmen des Machine Learning Operations Project im Sommersemester SS2024 durchgeführt wurde zur automatischen Generierung von professionellen Bewerbungsbildern aus hochgeladenen Nutzerfotos durch Nutzung der Text-/Bild-zu-Bild-Funktionalität von Stable Diffusion Modellen. Aus datenschutzrechtlichen Gründen wurden alle personenbezogenen Daten, sensible Informationen sowie proprietäre Rohdaten aus diesem Repository entfernt.

Die enthaltenen Skripte, Modelle und Strukturen spiegeln die technische Umsetzung wider, können jedoch ohne Originaldaten nicht vollständig reproduziert werden. 

Dieses Repository dient ausschließlich der Veranschaulichung des technischen Vorgehens (z. B. Datenvorverarbeitung, Modelltraining & -validierung, Visualisierung) und soll mein methodisches Vorgehen und die Struktur meiner Arbeit dokumentieren.


# MLOPS-Project 2024 - MLOps_SD

## Project Members, Mattermost Names and Campus Mail
- **Student A **
- **Student B**
- **Student C**
- **Thi Thuy Pham** 

## Project Goal
The project aims to utilize the Text/Image to Image capabilities of Stable Diffusion Models to enable end users to generate professional resume pictures based on their uploaded photos.


## Framework
We plan to use the [Diffusers framework](https://github.com/huggingface/diffusers) from Hugging Face and fine-tune using LoRA.

## Integration of Framework
We will leverage the Diffusers framework and use one of the provided pre-trained Stable Diffusion Text/Image to Image models, fine-tuning it on professional portrait pictures typically used for CVs or LinkedIn profiles. Users will be able to upload their own (portrait) pictures to generate suitable professional images. This can be achieved through the Image to Image function or by training a second LoRA with the user's pictures.

## Data
Initially, we do not expect to need a large amount of data, as we anticipate that a capable LoRA for this task can be trained with approximately 30-80 pictures. To obtain these images, we considered the following options:
- Using the Kaggle Dataset, focusing specifically on professional photos from different individuals.
- Searching for professional photos that are free to use on unsplash.com, Google Images, etc. We need about 50 different professional photos.
- Exploring large existing ML image datasets to see if they include suitable pictures.

## Deep Learning Models
Our intention is to use a pre-trained model from the stable diffusion model family, such as Stable Diffusion 1.5 or SDXL, and train and fine-tune the model using the LoRA method. We will start with the vanilla pre-trained models and test their performance to get an initial sense of their capabilities. After fine-tuning, we will select the model that best suits our task based on performance, computational resources, and time constraints.


## Project Organization
------------

    ├── LICENSE                              <- License for the project.
    ├── Makefile                             <- Makefile with commands like `make data` or `make train`
    ├── README.md                            <- The top-level README for developers using this project.
    ├── data
    │   ├── processed                        <- The final data sets for modeling.
    │   └── raw                              <- The original, immutable data dump.
    │
    ├── dockerfiles
    │   ├── predict_model.dockerfile         <- Dockerfile for prediction model.
    │   ├── train_model.dockerfile           <- Dockerfile for training model.
    │
    ├── docs                                 <- Documentation files with mkdoc
    │
    ├── environment.yml                      <- Conda environment configuration file
    │
    ├── mlops_project_2024                   <- Source code for use in this project.
    │   │
    │   ├── config                           <- Configuration files for different setups and experiments (hydra).
    │   │   ├── default_config.yaml          <- Default configuration.
    │   │   ├── Default_config_all.yaml      <- Default configuration used in the beginning of the project.
    │   │   └── example_config.yaml          <- Example configuration.
    │   │
    │   ├── data                             <- Scripts to handle data.
    │   │   └── make_dataset.py              <- Script to create datasets.
    │   │
    │   ├── models
    │   │   └── model.py                     <- Model definition script (not used due to pretrained model).
    │   │
    │   ├── predict_model.py                 <- Script to run model predictions with FastAPI endpoint.
    │   ├── train_model.py                   <- Script to train the model.
    │   │
    │   └── visualization                    <- Scripts for visualizing data and results (not used).
    │       └── visualize.py
    │
    ├── models
    │   └── pytorch_lora_weights.safetensors <- Trained LoRA finetuning file.
    │
    ├── notebooks
    │   └── inference_dev.ipynb              <- Notebook for model inference development.
    │
    ├── outputs                              <- Directory for storing output files.
    ├── pyproject.toml                       <- Project configuration file.
    ├── reports                              <- Project report.
    ├── requirements.txt                     <- The requirements file for reproducing the training and prediction environment.
    ├── requirements_dev.txt                 <- Additional requirements for development (not used).
    ├── requirements_test.txt                <- Requirements for running tests, CI and GitHub actions.
    │
    ├── test
    │   ├── test_api.py                      <- Tests using API endpoints.
    │   ├── test_data.py                     <- Tests for data processing scripts.
    │   └── test_model.py                    <- Tests for model scripts.
    │
    └── vertex_ai_train.yaml                 <- Configuration file for Vertex AI training.
