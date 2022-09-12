import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from sklearn.feature_selection import mutual_info_regression
#from sklearn.ensemble import RandomForestRegressor
#from xgboost import XGBRegressor
from tensorflow import keras

from keras import Sequential
from keras.layers import Dense, Dropout



def feature_importance(data, target='Share'):
    '''
    Calculate the importance of all data features against the target column using
    using the mutual information algorithm
    
    Args:
        data: Dataframe with relevent data
        target: Target column of data -- default at 'Share'
    
    Returns:
        mi_scores: Pandas series of all features and their mutial information score
        in descending order
    '''
    
    X = data.drop(target,axis=1)
    Y = data[target]
    
    mi_scores = mutual_info_regression(X, Y)
    mi_scores = pd.Series(mi_scores, name= "Scores", index=X.columns).sort_values(ascending=False)
    
    return mi_scores



def sort_features(data, fs, scores):
    '''
    Sort the feature columns of initial dataset by importance (based on mutual information score)
    
    Args:
        data: Dataframe with relevant data
        fs: Number of important features to keep
        scores: Pandas series of feature importance
        
    Returns:
        features: List with names of features to keep
        X_reduced: Dataframe that includes only the above features (all rows remain)
    '''
    
    
    scores_keep = scores[:fs]
    features = scores_keep.index
    X_reduced = data[features]
    
    return features, X_reduced


def plot_scores(scores, figsize, title):
    '''
    Create a horizontal bar plot
    
    Args:
        scores: Data to plot
        figsize: Size of the produced figure
        title: Title of figure
    '''
    
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(width, scores)
    
    for index, value in enumerate(scores):
        plt.text(value +0.005 , index, str(round(value,2)))
    
    plt.yticks(width, ticks)    
    plt.title(title)
    
    
def split_per_one_year(data, year, target = 'Share', scaling = False):
    
    '''
    Splits data into train and test datasets where test are the data for one year
    and rest is for training
    
    Args: 
        data: Dataframe for all years (historical data)
        year: Year that is used for test data
        target: Feature of data that will be the target for predictive algorithm -- default at Share
        scaling: Boolean to choose if we want the data to be scaled -- Default as False
        
    Returns:
        X_train: Training data without the actual values of target column
        Y_train: Labels of X_train
        X_test: Testing data without the actual values of target column
        Y_test: Labels of X_test
        
    '''
    
    to_drop_train = ['Rank','Player', 'Age', 'Tm', 'Year', 'Share']
    
    data2 = data.copy()
    
    train_data = data2[data2["Year"] != year]
    test_data = data2[data2["Year"] == year]
    
    Y_train = train_data[target]
    Y_test = test_data[target]
    
    train_data_num = train_data.drop(to_drop_train, axis=1)
    test_data_num = test_data.drop(to_drop_train, axis=1)
    
    cols = train_data_num.columns.values
    
    if scaling == True:
        scaler = StandardScaler()
        train_array = scaler.fit_transform(train_data_num)
        train_data2 = pd.DataFrame(train_array, index=train_data_num.index, columns=cols)
        train_data.loc[:,cols] = train_data2[cols]
        test_array = scaler.transform(test_data_num)
        test_data2 = pd.DataFrame(test_array, index=test_data_num.index, columns=cols)
        test_data.loc[:,cols] = test_data2[cols]

    
    X_train = train_data.drop(target, axis=1)
    #Y_train = train_data[target]
    
    
    X_test = test_data.drop(target, axis=1)
    #Y_test = test_data[target]
    
    
    return X_train, Y_train, X_test, Y_test



def run_model_one_year(regressor, data, year, scaling = False, NN = False, epochs = 50, batch_size = 50):
    '''
    Train and test a specific regressor using the split_per_one_year function to use one year
    as test data and the rest as training data. 
    
    Args:
        regressor: Regressor to train and predict the MVP standings for the chosen year
        data: Historical data with MVP candidates stats from past years
        year: Chosen year to test the trained model on
        scaling: Boolean to choose if data will be scaled -- default as False
        NN: Boolean, true if we use a NN model -- default as False
        epochs: number of epochs if NN=True -- default as 50
        batch_size: size of batches if NN=True -- default as 50
        
        
    Return:
        model: Trained regressor
        prediction: Dataframe of predicted win shares for each candidate that year with their stats
        summary: Dictionary with actual MVP, predicted MVP, MSE score and R2 score 
    '''
    
    to_drop_train = ['Rank','Player', 'Age', 'Tm', 'Year']
    
    model = regressor
    
    if (scaling == True) and (NN == True) :
        X_train, Y_train, X_test, Y_test = split_per_one_year(data, year, scaling=True)
    
        X_train_stats = X_train.drop(to_drop_train, axis=1)
        X_test_stats = X_test.drop(to_drop_train, axis=1)
        
        model.fit(X_train_stats, Y_train, epochs = epochs, batch_size = batch_size, verbose=0)

    else:
        X_train, Y_train, X_test, Y_test = split_per_one_year(data, year, scaling=scaling)
    
        X_train_stats = X_train.drop(to_drop_train, axis=1)
        X_test_stats = X_test.drop(to_drop_train, axis=1)
        
        model.fit(X_train_stats, Y_train) 
        
        
    Y_preds = model.predict(X_test_stats)
    
    MSE = mean_squared_error(Y_test.values, Y_preds)
    R2 = r2_score(Y_test.values, Y_preds)
    
    prediction = X_test.copy()
    prediction.insert(5, 'Actual Share', Y_test, True)
    prediction.insert(6, 'Predicted Share', Y_preds, True)
    
    predicted_winner = X_test['Player'].iloc[np.argmax(Y_preds)]
    
    summary = {"Year": year,
               "Actual MVP": X_test['Player'].iloc[0],
               "Predicted MVP": predicted_winner,
               "MSE score": MSE,
               "R2 score": R2
              }
    
    return model, prediction, summary


def run_model(regressor, data, years, scaling = False, NN = False, epochs = 50, batch_size = 50):
    '''
    Loop over all available years (except last year which is used for prediction) to train
    and test a regressor for each year using all other years as training data.
    
    Args:
        regressor: Regressor to train and predict MVP for each past year
        data: Historical data to use for training the model
        years: List of years included in data
        scaling: Boolean to choose if data will be scaled -- default as False
        NN: Boolean, true if we use a NN model
        epochs: number of epochs if NN=True
        batch_size: size of batches if NN=True        
        
    Returns:
        model: Trained regressor to use on final year data for predictions
        MVP_past_preds: List of predicted MVPs for past years
        MSE_scores: List of MSE scores
        R2_scores: List of R2 scores
    
    '''
 
    mvp_race_lst = []
    summaries = []
    results = []
    models = []
    
    for year in years:
        if NN == True:
            model, prediction, summary = run_model_one_year(regressor,
                                                            data,
                                                            year,
                                                            scaling=True,
                                                            NN=True,
                                                            epochs = epochs,
                                                            batch_size = batch_size
                                                            )
        else:
            model, prediction, summary = run_model_one_year(regressor,
                                                            data,
                                                            year,
                                                            scaling = scaling
                                                           )
        
        models.append(model)
        mvp_race_lst.append(prediction)
        summaries.append(summary)
        
        if summary['Actual MVP'] == summary['Predicted MVP']:
            res = 'Right'
        else:
            res = 'Wrong'
            
        results.append(res)
        
    summary_df = pd.DataFrame(summaries)
    summary_df['Result'] = results
        
    return models, mvp_race_lst, summary_df


def keras_model(n1, optimizer, learn_rate, dropout, act_hidden = 'relu', act_out = 'sigmoid'):
    '''
    Create a keras NN with a single hidden layer
    
    Args:
        n1: Number of neurons in first hidden layer
        n2: Number of neurons in second hidden layer
        optimizer: Optimization algorithm to be used
        learn_rate: Learning rate used by the optimizer
        dropout: Size of dropout
        activation_hidden: Activation dunction for hidden layers
        activation_out: Activation function for output layer
        
    Returns:
        model: Compiled Keras Sequential NN
    '''
    
    model = keras.Sequential()
    model.add(Dense(n1, input_shape=(14,), activation=act_hidden))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation=act_out))
    
    model.compile(loss = 'mse',
                  optimizer = optimizer(learning_rate = learn_rate),
                  metrics = ['mse']
                 )
    
    return model

def data_prep(train_data, test_data, feats, target = 'Share', scaling = False):
    '''
    Splitting the training and test datasets into X and Y scaling the data if desirable. For this project we     only scale the features and not the target since it's already an averaged quantity.
    
    Args:
        train_data: Training dataset
        test_data: Testing dataset
        feats: List of numeric features of both training and testing datasets
        target: Target feature -- default set to 'Share'
        scaling: Boolean indicating if we want to scale our data
        
    Returns:
        X_train: Training data without the actual values of target column
        Y_train: Labels of X_train
        X_test: Testing data without the actual values of target column
        Y_test: Labels of X_test        
    '''
    
    train_df = train_data.copy()
    test_df = test_data.copy()
    
    if scaling == True:
        feats_all = list(feats)
        
        features_train = train_df[feats_all]
        features_test = test_df[feats_all]
        
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(features_train)
        test_scaled = scaler.transform(features_test)
        
        train_df[feats_all] = train_scaled
        test_df[feats_all] = test_scaled
        
    X_train = train_df.drop(target, axis=1)
    Y_train = train_df[target]
    X_test = test_df.drop(target, axis=1)
    Y_test = test_df[target]
    
    return X_train, Y_train, X_test, Y_test