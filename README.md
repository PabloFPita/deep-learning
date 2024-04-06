
<img src="img/canonistia_logo.png" width="300">

# Canonist.ia - Labelling of real estate images

## Description
**Canonist.ia** is a project intended for the **classification of real estate images**. Applying **deep learning and transfer learning** techniques within the **PyTorch** framework, different models have been trained to be used via a **Streamlit** web application.

In the web application, users can upload images and the model will predict the category of the image. The project is particularly useful for real estate professionals and property owners.

Experiment tracking is done with **Weights and Biases** to monitor the performance of the models and to refine them.

## Installation
To install the required packages, run the following command:
```bash
pip install -r requirements.txt
```

## Usage
To **run the Streamlit web application**, execute the following command:
```bash
streamlit run app.py
```

To **train different models**, modify the training.ipynb notebook and execute it. Make sure to have a Weights and Biases account to log the experiments and add the **API key** to a .env file.