# Classification Of Criminal Recidivism Based On Machine learning Techniques

Application of Machine Learning to Predict and Classify criminals based on various parameters like Marital Status, Decile Score, Sex Code and other such attributes. 

- Criminal Recidivism is a tendency of a convicted felon to commit an offence again.
- This project aims at limiting and checking recidivism by classifying criminals into categories (Low, Medium, High) to gauge their tendency of commiting a crime.
- This is done according to the general trends of criminals with similar characteristics, taking into account various attributes like Sex Code, Marital Status, Ethnic Code, etc.
- We use and compare classification algorithms of Machine Learning like Random Forest classifiers, K-Nearest Neighbours algorithm and Logistic Regression for classification of the criminals into respective categories.
- Upon cleaning and filtering of the dataset, we obtained from Carnegie Mellon University (CMU) data repository, we implemented these algorithms using Scikit Learn library of python to perform machine learning on the data.
- Data mining and cleaning algorithms had to be applied to obtain the data in the desired format and also to get rid of the noisy and unwanted data.
- We observe how various algorithms perform and obtain the following results (in the OUTPUT_CMD.jpg file)-
  a] Random Forest Accuracry Score = 87.815 %
  b] kNN Accuracry Score = 86.88 %
  c] Logistic Regression Accuracry Score =  75.29 %

  Thus, the performance of the Random Forest classifier is the most suitable in classifying the criminals into Low, Medium and High score (risk) of recidivism.

- Further, we have plotted graphs, for data visualization of multiple attributes, which may be assumed to have no impact on the score, but are actually useful for classification.
- These graphs are saved as jpg images in the Analysis folder.

Future Scope and Conclusion :-

The results of classification should help police authorities to grant parole/bail to the criminals who are at a relatively lower risk of commiting another felony/crime.
This should also help reduce the crime rate in society, thus ensuring the welfare and harmony of the society.
The analysis provided through data visualisation may provide a better understanding and easier perspective/insight of what attributes contribute to most recidivism according to the trends of the 18,000(approxmiately) criminal records in the dataset.
