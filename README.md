# Skin Cancer Detection App



Here is a screenshot of the application:

![Application Screenshot](/screenshot.png)

## Description

The Skin Cancer Detection App helps users identify skin cancer by classifying images of skin lesions into 7 categories: Actinic Keratosis, Basal Cell Carcinoma, Benign Keratosis, Dermatofibroma, Melanoma, Melanocytic Nevi, Vascular Lesion. The application is powered by a TensorFlow model custom-built to accurately classify images. FastAPI is used to serve the model for a ReactJS web app.

## Setup for Python:

1. Install Python ([Setup instructions](https://wiki.python.org/moin/BeginnersGuide))

2. Install Python packages

```
pip3 install -r training/requirements.txt

```

## Setup for ReactJS

1. Install Nodejs ([Setup instructions](https://nodejs.org/en/download/package-manager/))
2. Install NPM ([Setup instructions](https://www.npmjs.com/get-npm))
3. Install dependencies

```bash
cd frontend
npm install --from-lock-json
npm audit fix
```


## Training the Model

1. Download the data from [kaggle](https://www.kaggle.com/datasets/sarthakj1997/german-garbage).
2. Run Jupyter Notebook in Browser.

```bash
jupyter notebook
```

3. Open `training/train.ipynb` in Jupyter Notebook.
4. Run all the Cells one by one.


## Running the API

### Using FastAPI

1. Get inside `api` folder

```bash
cd api
```

2. Run the FastAPI Server by running the `main.py` file


3. Your API is now running at `0.0.0.0:8081`

## Running the Frontend

1. Get inside `frontend` folder

```bash
cd frontend
```

2. Update `REACT_APP_API_URL` in `.env` to API URL if needed.
3. Run the frontend

```bash
npm run start
```


## Deploying the model on GCP

1. Create a [GCP account](https://console.cloud.google.com/freetrial/signup/tos?_ga=2.25841725.1677013893.1627213171-706917375.1627193643&_gac=1.124122488.1627227734.Cj0KCQjwl_SHBhCQARIsAFIFRVVUZFV7wUg-DVxSlsnlIwSGWxib-owC-s9k6rjWVaF4y7kp1aUv5eQaAj2kEALw_wcB).
2. Create a [Project on GCP](https://cloud.google.com/appengine/docs/standard/nodejs/building-app/creating-project) (Keep note of the project id).
3. Create a [GCP bucket](https://console.cloud.google.com/storage/browser/).
4. Upload the `garbage_InceptionV3.h5` model in the bucket in the path `models/garbage_InceptionV3.h5`.
5. Install Google Cloud SDK ([Setup instructions](https://cloud.google.com/sdk/docs/quickstarts)).
6. Authenticate with Google Cloud SDK.

```bash
gcloud auth login
```

7. Run the deployment script.

```bash
cd gcp
gcloud functions deploy predict --runtime python38 --trigger-http --memory 1024 --project project_id
```

8. Your model is now deployed.
