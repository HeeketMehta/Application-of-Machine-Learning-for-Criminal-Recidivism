# Classification Of Criminal Recidivism Based On Machine learning Techniques

```
Heeket Mehta, Shanay Shah, Neil Patel, Pratik Kanani (2020)<br />
“Classification of Criminal Recidivism Using Machine Learning Techniques”<br />
International Journal of Advanced Science and Technology, 29(04), pp. 5110<br />
http://sersc.org/journals/index.php/IJAST/article/view/24940 <br />
```

## Project Overview <br />

There are numerous cases in the recent times, where a criminal commits a crime, immediately after being granted parole, this is called Criminal Recidivism. The act of recidivism poses a great threat to the society and thus needs to be checked. <br />

This paper posits a machine learning approach to detect and predict the tendency of a criminal to commit recidivism. The proposed system helps classify the criminals into Low, Medium, and High risk of committing recidivism. Features like ‘Ethnic code’, ‘Marital Status’, ‘Age’, ‘Sex Code’, ‘Legal Status’ and many more are considered while training the model on the dataset. <br />

Supervised Classification Algorithms are implemented, and voting is subsequently done, to select the algorithm with the highest accuracy. The Random Forest Algorithm provides the highest accuracy score followed by KNN and lastly Logistic Regression. Moreover, the data is analyzed using visualization charts, where various attributes are deeply analyzed in relation to the target variable ‘Score Text’. <br />

Graphs between these attributes and the target variable highlight trends, which may provide useful insights to parole granting authorities while assessing a criminal for parole. Stratified K- Fold Cross Validation is used to bolster the results of the algorithms, which gives us accuracy score similar to the above algorithms. Thus, it validates and renders the algorithms unbiased and fair.<br /><br />

```
Concepts - Data Science, Data Analysis, Data Visualization, Machine Learning
Programming Language - Python
```

## Getting Started 

We mostly use Python in the project and hence, we used libraries that can be installed using - <br />
```
pip install pandas
pip install numpy
pip install seaborn
pip install matplotlib
pip install sklearn
```

## Proposed Architecture / Flow Diagram
We propose the flow diagram below and execute the methodology in the following sense, to obtain results of criminal recidivism, to understand various trends, and build machine learning models to understand which factors are the most significant and whether we can predict the tendency of criminal recidivism.<br />

![CR Model](https://github.com/HeeketMehta/Application-of-Machine-Learning-for-Criminal-Recidivism/blob/master/OUTPUTS/CR%20Model.png)<br />

## Exploratory Data Analysis

### Histogram of Age v/s Criminal Recidivism

![Age_vs_Reci](https://github.com/HeeketMehta/Application-of-Machine-Learning-for-Criminal-Recidivism/blob/master/OUTPUTS/Histogram%20of%20AGE.png)<br />

### Ethnic Code v/s Criminal Recidivism

![ethnic_vs_Reci](https://github.com/HeeketMehta/Application-of-Machine-Learning-for-Criminal-Recidivism/blob/master/OUTPUTS/Ethnic%20Code.JPG)<br />

### Legal Status v/s Criminal Recidivism

![legal_status_vs_Reci](https://github.com/HeeketMehta/Application-of-Machine-Learning-for-Criminal-Recidivism/blob/master/OUTPUTS/Legal%20Status.JPG)<br />


### Marital Status v/s Criminal Recidivism

![Marital_status_vs_Reci](https://github.com/HeeketMehta/Application-of-Machine-Learning-for-Criminal-Recidivism/blob/master/OUTPUTS/Marital%20Status.JPG)<br />


### Importance of Attributes

![Attr_importance](https://github.com/HeeketMehta/Application-of-Machine-Learning-for-Criminal-Recidivism/blob/master/OUTPUTS/Importance%20of%20Attributes.JPG)<br />




## Machine Learning Results

On comparing KNN, Logistic Regression and Random Forest, we get the following results of categorising or figuring if a person with given set of attributes would commit recidivism or not, and with how much probability.
![Compare_vals](https://github.com/HeeketMehta/Application-of-Machine-Learning-for-Criminal-Recidivism/blob/master/OUTPUTS/OUTPUT_CMD.JPG)<br />


## Conclusion
We have done a thorough studny on tendency of criminal recidivism and made a ML model to predict the likelihood of someone commiting crimal recidivism.
Please check out the paper we published at the following URL - 
```
http://sersc.org/journals/index.php/IJAST/article/view/24940 <br />
```
We really appreciate your interest

## Authors
```
Heeket Mehta
Shanay Shah
Neil Patel
```

## Mentorship -
```
Prof. Pratik Kanani
```





