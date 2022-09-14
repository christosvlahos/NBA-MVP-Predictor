# NBA-MVP-Predictor
A data science project, predicting last year's NBA MVP using past data and multiple predictive models


# NBA-MVP-Predictor
Data Science project predicting last season's NBA MVP based on past data, using multiple models.

## Data
The data have been acquired from the basketball-reference website using the pandas HTML table scraper.

The dataset contains all per-game and advanced statistics data for the MVP candidates from 1980 till 2022. The data of the last year have been used strictly for testing purposes to ensure our models are as unbiased as possible.

The code to scrape the per-36-minutes and per-100-possessions data is there commented out for potential future use.

## Notebooks
The notebooks folder includes three jupyter notebooks focused on 
1. Data scraping
2. MVP predictions and exploratory data analysis
3. Hyperparameter tuning

There is also a python script including all the custom functions used in the previously mentioned notebooks.

## Figures
Folder including interactive scatter plots from the EDA section of the MVP_predictions notebook, showing the relation of the share of MVP votes of each candidate with some main features.

## Environment
The run the files in this repository you can create a virtual environment using the requirements.txt file to install the necessary dependencies.
