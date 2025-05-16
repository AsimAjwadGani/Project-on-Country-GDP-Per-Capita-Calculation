CSE422 Project Report
        Project on Country GDP Per Capita Calculation


Team Members (CSE422 Section 04):

Asim Ajwad Gani (ID: 21201083)

Mohammad Nazibul Bashar Chowdhury (ID: 21301736)






Table of Contents:


Introduction: pg#3


Dataset Description: pg#4


Dataset Pre-processing: pg#6


Feature Scaling: pg#7


Dataset Splitting: pg#7


Model Training and Testing: pg#8


Model Selection/Comparison Analysis: pg#10


Conclusion: pg#11












1) Introduction: 

Gross Domestic Product, GDP,is a key economic indicator used to measure the overall health of a country's economy. GDP is the total monetary or market value of all the available services and finished goods within the country in a certain time period, and is typically calculated on an annual basis. The GDP of a country provides a comprehensive economic overview. It is used to estimate the size of the economy, and its growth rate; in short, its overall economic health. The GDP per capita denotes a country’s economic output per person, and this is an important metric used to determine how economically prosperous a country is. 

Predicting future GDP growth is important for a nation that needs to make informed decisions about resource allocation and to plan its economies.  By developing accurate prediction models for GDP growth and predicting future GDP per capita values, we can help lawmakers, investors, and other stakeholders make informed decisions about economic development and contribute to the overall stability and prosperity of the global economy.

We used several Python libraries for our project, namely Pandas and Numpy (for making data frames and managing the data), Matplotlib, Seaborn, Plotly (for our visualization tools) and most importantly to train and use machine learning models from Scikit-Learn. 

In this project, we aim to develop models for predicting the GDP per capita growth of different countries worldwide. We will use GDP and GDP per capita data over the years (from 1960-2020) to train and evaluate our models, as we believe these variables are the most important factors in predicting GDP growth. We aim to identify the most important predictors of GDP growth and develop models that can be used to predict GDP in different countries as accurately as possible. Through this project, we aim to provide insights into the factors that influence the GDP of a nation and identify proactive steps that can be taken to promote sustainable economic growth globally.

2) Dataset Description:

Dataset source: 

World GDP(GDP, GDP per capita, and annual growths) (kaggle.com)

World GDP(GDP, GDP per capita, and annual growths)(by OZGUR CEM TAS) 
(GDP and GDP per capita for the years 1960-2020; Data Source: World Bank)


Column headers/Features, Label, Number of instances/Rows Information:

Our GDP dataset includes five key features: Country Name, Country Code, year, GDP, and GDP per capita. 


Types of features: There are in total 5 features; two categorical (Country Name and Country Code), and three numeric features (year, GDP, and GDP per capita). 

Correlation of the numerical features(using heatmap):



Imbalanced/Biasness: Values in the dataset were quite balanced, with no particular skew for any feature. Values were not repeated for features, so we can say data was not biased.  However, there were missing values of GDP and GDP per capita.
These missing values were imputed by the same values (based on the median of the GDP or GDP per capita of the country which was input by the user).     




3) Dataset Pre-processing:
Both our dataframes consisted of null values from the features GDP and GDP per capita. After merging them we used the median of each feature to replace the null values. Simple imputer was imported from sklearn.impute for this job
Both files had a redundant column named ‘unnamed : 65’. We dropped it from both data frames before merging them
Two categorical features were present, Country Name and code which were simply ignored
Our dataframe not only had single countries, but also regions too such as ‘Arab World’, ‘Early-demographic dividend’, etc. Recurring country data were also present. To tackle both these issues, we made a new dataframe and manually entered each country's name that we wanted by matching it with the existing one.





   

4) Feature Scaling(as required):
 
In the code, Min-Max scaling is applied to the GDP values using the MinMaxScaler from Scikit-learn. The GDP values are scaled to the range between 0 and 1. After scaling, the GDP values are used as the target variable for training linear regression, decision tree, and random forest models. Decision tree and random forest models typically do not require feature scaling but all three models (Linear Regression, Decision Tree, and Random Forest) utilize feature scaling using Min-Max scaling to ensure that all input features are on a similar scale.






5) Dataset Splitting:

Scikit Learn’s train_test_split method was used to train and test the models. 
For all the models, the ‘year’ feature in the dataset was set as the x variable. The GDP feature was set as the y variable.

The x and y variables were then trained, and compared against the testing data. The testing data is the part of the data which is not used for training, and as a result the model is unfamiliar with the data as it is not trained with this. The testing data is used to compare the training and testing accuracy of the trained model. All the models were trained by setting the parameter to random_state. This was done so that the training data and testing data did not vary from time to time, and remain fixed.

For the Linear Regression Model, the Decision Tree Model and Random Forest model, the test_size was set to 30%, which means that 30% of the total data would be used to test the model. The remaining 70% would be used to train the model.



6) Model training and testing:

In our project, we used three data models to predict GDP; Linear Regression, Decision Tree regression, and Random Forest regression. 

To evaluate the performance of the models, we used two metrics: mean squared error (MSE) and root mean squared error (RMSE). 
For MSE, the difference between the actual data point and the regression line (predicted data point) is measured (the error). The error for each of the data points is squared, and all the values of the errors are summed. RMSE is the square root of MSE.

	       Mean Squared Error (MSE)= Summation of (individual errors^2)
	       Root Mean Squared Error (RMSE)= √(MSE)

For a third metric, we also used R2 score (Coefficient of Determination) to determine the accuracy of our models. 

Linear Regression is a model that predicts the value of a dependent variable based on one or more independent variables. It plots a best fit line that fits the data points so that we can use it to make predictions. Linear regression assumes that the relationship between the two variables is linear and uses a method to minimize the difference between predicted and actual values, and plots the predicted (best fit line). For this model we set the test size to be 30%, and training size to be 70% respectively.

Taking the input country to be Bangladesh (for all models), we got a R2 score of 71.51% for the linear regression model. Training and testing accuracy was found to be 61.75% and 71.52%. 


Decision Tree Regression is a model that observes the features of the data and trains the model in the structure of a tree to predict the data. To create this tree, the model splits the dataset into smaller groups based on the values of the feature, and this process is repeated recursively until the model can make accurate predictions about the data. To make the model perform better, we can tune some of its settings. The parameter that we modified to tune our model is the max depth of the tree. Initially, we set it to 20, and after iterating with multiple depths, setting the max depth to 10, which gave us the best results comparatively. 

R2 score was found to be 98.91%. Training and testing accuracy was found to be 100% and 98.91% respectively. 


Random Forest Regression is a model used for predicting target values based on the input variables. It works by creating multiple decision trees that are each trained on a subset of the input data. The model then combines the predictions of all the individual trees to come up with the final prediction. With Random Forest, the R2 score observed was 99.36%. Training accuracy was found to be 99.82%, and testing accuracy was 99.36%. 

For all our models we used both unscaled and scaled data. Though using scaled data did not show very significant improvements in most models.



7) Model selection/Comparison Analysis:





Training and testing accuracy was mostly the highest for Random Forest Regression and Decision Tree Regression models. Linear Regression had the least accuracy percentages throughout most of our trials. 




8) Conclusion:

In our project, we tested three different regression models to predict the GDP.  All the models we used in the project work in a different mechanism, but the purpose of all of them here was to calculate GDP. Every model could do that in their own way. We used the same three metrics to count accuracy percentages (R2 score, MSE and RMSE) for all models, so that they all can be compared on the same scale. 

We tried running the models with a different input country, and in most of our trials the Random Forest regression produced the best results, followed closely by Decision Tree Regression with higher accuracy scores for both training and testing data sets compared to the other models. Therefore, we can conclude that Random Forest regression is the most suitable model for predicting the GDP based on the features we used in this project.
# Project-on-Country-GDP-Per-Capita-Calculation
