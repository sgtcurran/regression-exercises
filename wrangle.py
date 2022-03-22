#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from scipy.stats import chi2_contingency, ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, RocCurveDisplay


import seaborn as sns
from copy import deepcopy

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import RidgeClassifier
#%%
classifications_list = [
    LogisticRegression(max_iter=500), 
    KNeighborsClassifier(n_neighbors=7), DecisionTreeClassifier(max_depth=3), RandomForestClassifier(n_estimators=1000, criterion='entropy',max_features='sqrt'),
    ExtraTreesClassifier(), AdaBoostClassifier(), GradientBoostingClassifier(learning_rate=0.01, loss='exponential', max_depth=3, n_estimators=1000, subsample=0.5), 
    MLPClassifier(activation='logistic', alpha=0.0001, learning_rate='constant', solver='adam'),
    RidgeClassifier(alpha=0.4)
]

#%%
classifierList = [
    LogisticRegression(max_iter=500), 
    KNeighborsClassifier(n_neighbors=7), DecisionTreeClassifier(max_depth=3), RandomForestClassifier(n_estimators=1000, criterion='entropy',max_features='sqrt'),
    ExtraTreesClassifier(), AdaBoostClassifier(), GradientBoostingClassifier(learning_rate=0.01, loss='exponential', max_depth=3, n_estimators=1000, subsample=0.5), 
    MLPClassifier(activation='logistic', alpha=0.0001, learning_rate='constant', solver='adam'),
    RidgeClassifier(alpha=0.4)
]
#%%

#%%
def model_magic(X_train, y_train, X_validate, y_validate, X_test, y_test, classifierList):
    """
    Description
    ----
    This function tests out all models listed in classifierList.
    
    If the model isn't valid, the function prints out the invalid
    name of the model along with it's error.
    The accuracy score, confusion matric, positive precision, recall 
    and f-scores, and negative prevision, recall and f-scores.
    
    Data is then appended into a dictionary with columns.
    
    
    Parameters
    ----
    The X_train split for the dataframe.
        
    The X_test split for the dataframe.

    The X_validate split for dataframe.

    The y_validate splot for dataframe.

    The y_train split for the dataframe.
        
    The y_test split for the dataframe.
        
    classifierList (list of models):
        The list of models chosen.
    
    Returns
    ----
    dic (dataframe):
        A dataframe of dic.
        
    """
    # Cretate Dictionary for model stats train, validate, test 
    dic = {'ModelName': [], 'AccuracyScore': [], 'AccuracyScoreVAL': [],
           'CorrectPredictionsCount': [], 'CorrectPredictionsCountVAL': [], 'Total': [], 'TotalVAL': [], 
           'PosPrecScore': [], 'PosPrecScoreVAL':[], 'PosRecScore': [], 'PosRecScoreVAL': [] ,'PosFScore': [],
           'PosFScoreVAL': [],'NegPrecScore': [], 'NegPrecScoreVAL': [] ,'NegRecScore': [], 'NegRecScoreVAL': [],
           'NegFScore': [], 'NegFScoreVAL': [], 'TNPercentage': [], 'TNPercentageVAL': [],'TPPercentage': [], 
           'TPPercentageVAL': [],'FNPercentage': [], 'FNPercentageVAL': [], 'FPPercentage': [], 'FPPercentageVAL': []}
    
    # Deepcopy the classifierList
    models = deepcopy(classifierList)
    
    # Test each models in the list to verify 
    for i in range(len(classifierList)):
        try:
            model = classifierList[i]
            model.fit(X_train, y_train)
        except Exception as e:
            print("==============================================================")
            print(f"I wasn't able to score with the model: {classifications_list[i]}")
            print(f"This was the error I've received from my master:\n\n{e}.")
            print("\nI didn't let it faze me though, for now I've skipped this model.")
            print("==============================================================\n")
            models.remove(classifierList[i]) # Remove invalid models from list
    
    # Loop through all models
    for classifier in range(len(models)):
        # removes any 
        modelName = re.sub(r"\([^()]*\)", '', str(models[classifier]))
        # Performance
        model = models[classifier]
        model.fit(X_train, y_train)          
        pred = model.predict(X_test)
        pred1 = model.predict(X_validate)
        # Results
        acc_score = accuracy_score(y_test, pred)
        acc_score1 = accuracy_score(y_validate, pred1) 
        noOfCorrect = accuracy_score(y_test, pred, normalize = False)
        noOfCorrect1 = accuracy_score(y_validate, pred1, normalize = False) 
        total = noOfCorrect/acc_score
        total1 = noOfCorrect1/acc_score1
        Confusing = confusion_matrix(y_test, pred)
        madConfusing1 = confusion_matrix(y_validate, pred1)
        # calculations 
        dpps = Confusing[1][1] / (Confusing[1][1] + Confusing[0][1]) # pos prec score
        dpps1 = madConfusing1[1][1] / (madConfusing1[1][1] + madConfusing1[0][1])
        dprs = Confusing[1][1] / (Confusing[1][1] + Confusing[1][0]) # pos rec score
        dprs1 = madConfusing1[1][1] / (madConfusing1[1][1] + madConfusing1[1][0])
        dpfs = 2 * (dpps * dprs) / (dpps + dprs) # pos f1 score
        dpfs1 = 2 * (dpps1 * dprs1) / (dpps1 + dprs1) # pos f1 score
        dnps = Confusing[0][0] / (Confusing[0][0] + Confusing[1][0]) # neg prec score
        dnps1 = madConfusing1[0][0] / (madConfusing1[0][0] + madConfusing1[1][0])
        dnrs = Confusing[0][0] / (Confusing[0][0] + Confusing[0][1]) # neg rec score
        dnrs1 = madConfusing1[0][0] / (madConfusing1[0][0] + madConfusing1[0][1])
        dnfs = 2 * (dnps * dnrs) / (dnps + dnrs) # neg f1 score
        dnfs1 = 2 * (dnps1 * dnrs1) / (dnps1 + dnrs1) 
               

        # Save Calulations and append to dictionary 
        dic['ModelName'].append(modelName)
        dic['AccuracyScore'].append(acc_score)
        dic['AccuracyScoreVAL'].append(acc_score1)
        dic['CorrectPredictionsCount'].append(noOfCorrect)
        dic['CorrectPredictionsCountVAL'].append(noOfCorrect1)
        dic['Total'].append(total)
        dic['TotalVAL'].append(total1)
        dic['PosPrecScore'].append(dpps)
        dic['PosPrecScoreVAL'].append(dpps1)
        dic['PosRecScore'].append(dprs)
        dic['PosRecScoreVAL'].append(dprs1)
        dic['PosFScore'].append(dpfs)
        dic['PosFScoreVAL'].append(dpfs1)
        dic['NegPrecScore'].append(dnps)
        dic['NegPrecScoreVAL'].append(dnps1)
        dic['NegRecScore'].append(dnrs)
        dic['NegRecScoreVAL'].append(dnrs1)
        dic['NegFScore'].append(dnfs)
        dic['NegFScoreVAL'].append(dnfs1)
        dic['TNPercentage'].append(Confusing[0][0]/total*100)
        dic['TNPercentageVAL'].append(madConfusing1[0][0]/total*100)
        dic['TPPercentage'].append(Confusing[1][1]/total*100)
        dic['TPPercentageVAL'].append(madConfusing1[1][1]/total*100)
        dic['FNPercentage'].append(Confusing[1][0]/total*100)
        dic['FNPercentageVAL'].append(madConfusing1[1][0]/total*100)
        dic['FPPercentage'].append(Confusing[0][1]/total*100)
        dic['FPPercentageVAL'].append(madConfusing1[0][1]/total*100)
        
    return pd.DataFrame.from_dict(dic)


#%%
def corrstatsgraphs(df):
    """
    Description
    ----
    Outputs the general statistical description of the dataframe,
    outputs the correlation heatmap with target label, and outputs a distribution plot.
    
    Parameters
    ----
    df(DataFrame):
        The dataframe for which information will be displayed.
        
    Returns
    ----
    useful stats, correlation, and subplots
    
    """
    # Description
    print("Descriptive Stats:")
    display(df.describe().T)
    
    # Heatmap with min -1 to max 1 to all variables
    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html
    corr = df.corr()
    f, ax = plt.subplots(figsize=(22, 17))
    plt.title("Heatmap", fontsize = 'x-large')
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 21, as_cmap=True)
    sns.heatmap(corr, annot=True, mask = mask, cmap=cmap
    )
    # Correlation Heatmap with min -1 to max 1 in conjuction with pd.corr 
    plt.figure(figsize=(10, 8)) 
    plt.title("Heatmap", fontsize = 'x-large')
    sns.heatmap(df.corr()[['taxvaluedollarcnt']].sort_values(by='taxvaluedollarcnt', 
    ascending=False), vmin=-1, vmax=1, annot=True, cmap='BrBG'
    )
    # Correlation Heatmap with min -1 to max 1 in conjuction with pd.corr
    plt.figure(figsize=(16,10))
    df.corr()['taxvaluedollarcnt'].sort_values(ascending=False).plot(kind='bar', figsize=(20,5), cmap='BrBG'
    )

    
#%%
#%%
def magic2(edf):
    """
    Description
    ----
    Splits a single given dataframe using 
    'train_test_split'
    
    
    Parameters
    ----
    df (dataframe):
        The dataframe to use for modeling.
    
    test_size (float):
        The test_size that you want to give for 
        train_test_split.
        The default test_size is set to 0.2 for 
        test_validate and test
        The default test_size is set to 0.3 for
        train and validate
    
    Returns
    ----
    save (dataframe):
        The dataframe after running the splits in
        `model_magic`
        
    """
    
    # Split
    train_validate, test = train_test_split(edf, test_size=0.2, random_state=42, stratify=edf['churn'])
    train, validate = train_test_split(train_validate, test_size=0.3, random_state=42, stratify=train_validate['churn'])
    # get dummmys 
    dummy_train = pd.get_dummies(train[['gender', 'partner', 'dependents', 'phone_service', 'multiple_lines', 
                            'online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 
                            'streaming_movies', 'paperless_billing', 'churn']], drop_first=[True])
    dummy_validate = pd.get_dummies(validate[['gender', 'partner', 'dependents', 'phone_service', 'multiple_lines', 
                            'online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 
                            'streaming_movies', 'paperless_billing', 'churn']], drop_first=[True])
    dummy_test = pd.get_dummies(test[['gender', 'partner', 'dependents', 'phone_service', 'multiple_lines', 
                            'online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 
                            'streaming_movies', 'paperless_billing', 'churn']], drop_first=[True])
    #merge dummies with orginal dataframe
    train = pd.concat([train, dummy_train], axis=1)
    validate = pd.concat([validate, dummy_validate], axis=1)
    test = pd.concat([test, dummy_test], axis=1)
    #drop columns with corresponding dummies
    train = train.drop(columns=['gender', 'partner', 'dependents', 'phone_service', 'multiple_lines', 
                            'online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 
                            'streaming_movies', 'paperless_billing', 'churn'])
    validate = validate.drop(columns=['gender', 'partner', 'dependents', 'phone_service', 'multiple_lines', 
                            'online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 
                            'streaming_movies', 'paperless_billing', 'churn'])
    test = test.drop(columns=['gender', 'partner', 'dependents', 'phone_service', 'multiple_lines', 
                            'online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 
                            'streaming_movies', 'paperless_billing', 'churn'])
    # assign x, y trian, validate, test 
    X_train = train.drop(columns=['customer_id','churn_Yes'])
    y_train = train.churn_Yes

    X_validate = validate.drop(columns=['customer_id','churn_Yes'])
    y_validate = validate.churn_Yes

    X_test = test.drop(columns=['customer_id','churn_Yes'])
    y_test = test.churn_Yes

    save = model_magic(X_train, y_train, X_validate, y_validate, X_test, y_test, classifierList)

    
    
    return save
#%%
def prep_zillow(pdf):
    """
    Converts to_numeric, drops rows with NaN values, splits data using sklean.train_test_split, hot encodes with pd.get_dummies(),
    concats dataframe with dummies, drops original dummies, and assigns X, y variables to train, validate, and test.

    Returns:
    X_train, y_train, X_validate, y_validate, X_test, y_test
    
    """
    # convert total_charges to float64 error will be NaN
    pdf['total_charges'] = pd.to_numeric(pdf['total_charges'], errors='coerce')
    
    # Drop NaN rows since they are 0 in total_charges and reset_index to 0 
    pdf = pdf.dropna(axis=0)
    pdf.reset_index(drop=True)

    # Split data into train, validate, test 
    train_validate, test = train_test_split(pdf, test_size=0.2, random_state=42, stratify=pdf['churn'])
    train, validate = train_test_split(train_validate, test_size=0.3, random_state=42, stratify=train_validate['churn'])
    # hot encoding using pd.get_dummies for non-numbrical catagorical data for train, validate, test
    dummy_train = pd.get_dummies(train[['gender', 'partner', 'dependents', 'phone_service', 'multiple_lines', 
                            'online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 
                            'streaming_movies', 'paperless_billing', 'churn']], drop_first=[True])
    dummy_validate = pd.get_dummies(validate[['gender', 'partner', 'dependents', 'phone_service', 'multiple_lines', 
                            'online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 
                            'streaming_movies', 'paperless_billing', 'churn']], drop_first=[True])
    dummy_test = pd.get_dummies(test[['gender', 'partner', 'dependents', 'phone_service', 'multiple_lines', 
                            'online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 
                            'streaming_movies', 'paperless_billing', 'churn']], drop_first=[True])
    # concat pdf dataframe with dummies 
    train = pd.concat([train, dummy_train], axis=1)
    validate = pd.concat([validate, dummy_validate], axis=1)
    test = pd.concat([test, dummy_test], axis=1)
    # drop original columns 
    train = train.drop(columns=['gender', 'partner', 'dependents', 'phone_service', 'multiple_lines', 
                            'online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 
                            'streaming_movies', 'paperless_billing', 'churn'])
    validate = validate.drop(columns=['gender', 'partner', 'dependents', 'phone_service', 'multiple_lines', 
                            'online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 
                            'streaming_movies', 'paperless_billing', 'churn'])
    test = test.drop(columns=['gender', 'partner', 'dependents', 'phone_service', 'multiple_lines', 
                            'online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 
                            'streaming_movies', 'paperless_billing', 'churn'])
    
    return train, validate, test
#%%
# CHI-SQUARED TEST FUNCTION to idenify non-corralations between target

def chi2_test(train, target, cat_features):

    '''
    Loop function to create pd.crosstab of target and features that correlates & not correlates with target.
    If it does not correlates to append to new list call Remove.

    '''
    # list of non-corralations features to be removed from X_train, X_validate, X_test
    Remove=[]
    print('The Chi2_test result are : \n')
    for feature in cat_features: 
        CrossResult=pd.crosstab(index = train[target], columns=train[feature])
        # p-value is index [1]
        pval = chi2_contingency(CrossResult)[1]
        # print(Result)
        # If the ChiSquare P-Value is <0.05, that means we reject H0
        if (pval < 0.05):
            print(feature, 'correlates with', target, '| P-Value:', pval)
        else:
            print(feature, 'does not correlates with', target, '| P-Value:', pval) 
            # append to remove list  
            Remove.append(feature)     
    print("\n\n")
    # return list 
    return(Remove)
#%%%
## PEARSONR CORRELATION TEST FUNCTION
def correlation_test(df, target, num_features):
    '''
    Given two subgroups from a dataset, conducts a correlation test for linear relationship between df and target.
    Utilizes the method provided in the Codeup curriculum for conducting correlation test using
    scipy and pandas. 
    '''
    # list of non-corralations features to be removed from X_train, X_validate, X_test
    Remove=[]

    print('The T-Test result are :\n')
    for feature in num_features:
        feature_num = df.groupby(target)[feature].apply(list)
        # p-value is index [1]
        pvalue = ttest_ind(*feature_num)[1]
        # If the T-Test P-Value is <0.05
        if (pvalue < 0.05):
            print(feature, 'correlates with', target , '| P-Value:', pvalue)
        else:
            print(feature, ' NOT correlates with', target , '| P-Value:', pvalue)
            Remove.append(feature)
    print("\n\n")
    # returns list 
    return(Remove)
#%%%
def X_y(train, validate, test):

    # drop target data for X_train and assign target data for y_train
    X_train = train.drop(columns=['taxvaluedollarcnt'])
    y_train = train.taxvaluedollarcnt
    # drop target data for X_validate and assign target data for y_validate
    X_validate = validate.drop(columns=['taxvaluedollarcnt'])
    y_validate = validate.taxvaluedollarcnt
    # drop target data for X_test and assign target data for y_test 
    X_test = test.drop(columns=['taxvaluedollarcnt'])
    y_test = test.taxvaluedollarcnt
    
    return X_train, y_train, X_validate, y_validate, X_test, y_test

#%%
def wrangle_grades(df):

    """
    Read student_grades csv file into a pandas DataFrame,
    drop student_id column, replace whitespaces with NaN values,
    drop any rows with Null values, convert all columns to int64,
    return cleaned student grades DataFrame.
    """
    # Acquire data from csv file.
    grades = pd.read_csv("student_grades.csv")
    # Replace white space values with NaN values.
    grades = grades.replace(r"^\s*$", np.nan, regex=True)
    # Drop all rows with NaN values.
    df = grades.dropna()
    # Convert all columns to int64 data types.
    df = df.astype("int")
    return df
#%%
def X_full_y_full(df):
    # used to split train to figure out the best imputer method 
    # drop target data for X_train and assign target data for y_train
    X_full = df.drop(columns=['id','propertylandusetypeid'], axis=1)
    y_full = df.drop(columns=['id','propertylandusetypeid'])
    
    
    return X_full, y_full