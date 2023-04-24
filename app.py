#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import openpyxl

st.title('Wine Quality Prediction')
st.subheader('**Quality Index:**')
st.markdown('* **1-5 = Bad**')
st.markdown('* **6-10 = Good**')

dataset = st.selectbox('Select Wine type', ('Red Wine', 'White Wine'))

def get_data(dataset):
    data_red = pd.read_excel('Data/winequality-red.xlsx')
    data_white = pd.read_excel('Data/winequality-white.xlsx')
    if dataset == 'Red Wine':
        data = data_red
    else:
        data = data_white
    return data

data_heatmap= get_data(dataset)
data= get_data(dataset)
def get_dataset(dataset):
    bins = (1, 5, 10)
    groups = ['1', '2']
    data['quality'] = pd.cut(data['quality'], bins=bins, labels=groups)
    x = data.drop(columns=['quality'])
    y = data['quality']
    return x, y

x, y = get_dataset(data)
st.write('Shape of dataset:', data.shape)


with st.beta_expander('Data Visualisation'):
    plot = st.selectbox('Select Plot type', ('Histogram', 'Box Plot', 'Heat Map'))

    if plot=='Heat Map':
        fig1=plt.figure(figsize=(8,6))
        heatmap = sns.heatmap(data_heatmap.corr()[['quality']].sort_values(by='quality', ascending=False), vmin=-1,
                              vmax=1, annot=True)
        heatmap.set_title('Features Correlating with quality', fontdict={'fontsize': 18}, pad=16)
        st.pyplot(fig1)
    else:
        feature = st.selectbox('Select Feature', ('fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                                                  'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                                                  'pH', 'sulphates', 'alcohol'))
        if plot == 'Histogram':
            fig2 = plt.figure(figsize=(7, 5))
            plt.xlabel(feature)
            sns.distplot(x[feature])
            st.pyplot(fig2)
        else:
            fig3 = plt.figure(figsize=(3, 3))
            plt.xlabel(feature)
            plt.boxplot(x=x[feature])
            st.pyplot(fig3)



with st.beta_expander('Prediction'):


    classifier = st.selectbox('Select Classifier', ('KNN', 'SVM', 'Random Forest', 'XGBoost'))


    # In[24]:

    def get_algorithm(classifier):
        params = dict()
        if classifier == 'KNN':
            n_neighbors = st.slider('n_neighbors', 1, 20)
            params['n_neighbors'] = n_neighbors
        elif classifier == 'SVM':
            C = st.slider('C', 1.0, 15.0)
            params['C'] = C
            kernel = st.selectbox('Select the kernel type', ('linear', 'poly', 'rbf', 'sigmoid'))
            params['kernel'] = kernel
        elif classifier == 'Random Forest':
            n_estimators = st.slider('n_estimators', 100, 1000)
            max_depth = st.slider('max_depth', 1, 15)
            params['n_estimators'] = n_estimators
            params['max_depth'] = max_depth
        else:
            learning_rate = st.slider('learning_rate', 0.001, 0.5)
            max_depth = st.slider('max_depth', 1, 15)
            params['learning_rate'] = learning_rate
            params['max_depth'] = max_depth
        return params


    # In[25]:

    params = get_algorithm(classifier)

    # In[31]:

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler


    # In[32]:

    def model_update(classifier, params):
        if classifier == 'KNN':
            model = KNeighborsClassifier(n_neighbors=params['n_neighbors'])
        elif classifier == 'SVM':
            model = SVC(C=params['C'], kernel=params['kernel'])
        elif classifier == 'Random Forest':
            model = RandomForestClassifier(max_depth=params['max_depth'], n_estimators=params['n_estimators'])
        else:
            model = XGBClassifier(learning_rate=params['learning_rate'], max_depth=params['max_depth'])
        return model


    model = model_update(classifier, params)

    # In[34]:

    sc = StandardScaler()
    x = sc.fit_transform(x)

    # In[35]:

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)


    # In[36]:

    model.fit(x_train, y_train)

    y_predict = model.predict(x_test)
    accuracy1 = accuracy_score(y_test, y_predict)
    accuracy1 = accuracy1 * 100
    accuracy1 = round(accuracy1, 2)
    st.write(f'Accuracy is {accuracy1}%')


    st.write('**If you want to make quality prediction for custom input values, enter the values below**')
    #if st.button('Custom Prediction'):
    def user_input_features():
        fixed_acidity = st.number_input('Fixed Acidity')
        volatile_acidity = st.number_input('Volatile Acidity')
        citric_acid = st.number_input('Citric Acid')
        residual_sugar = st.number_input('Residual Sugar')
        chlorides = st.number_input('Chlorides')
        free_sulfur_dioxide = st.number_input('Free sulfur dioxide')
        total_sulfur_dioxide = st.number_input('Total sulfur dioxide')
        density = st.number_input('Density')
        pH = st.number_input('pH')
        sulphates = st.number_input('Sulphates')
        alcohol = st.number_input('Alcohol')

        to_predict = {'fixed_acidity': fixed_acidity, 'volatile_acidity': volatile_acidity,
                      'citric_acid': citric_acid, 'residual_sugar': residual_sugar,
                      'chlorides': chlorides, 'free_sulfur_dioxide': free_sulfur_dioxide,
                      'total_sulfur_dioxide': total_sulfur_dioxide, 'density': density, 'pH': pH,
                      'sulphates': sulphates, 'alcohol': alcohol}
        df = pd.DataFrame(to_predict, index=[0])
        return df


    df = user_input_features()
    def quality_prediction():
        if classifier == 'XGBoost':
            names = model.get_booster().feature_names
            df.columns = names
            y_custom_predict = model.predict(df)
            if y_custom_predict == 1:
                text = '**Your Wine is of Bad quality**'
            else:
                text = '**Your Wine is of Good quality**'
        else:
            y_custom_predict = model.predict(df)
            if y_custom_predict == 2:
                text = '**Your Wine is of Good quality**'
            else:
                text = '**Your Wine is of Bad quality**'
        return text
    text= quality_prediction()

    if st.button('Predict'):
        st.write(text)
