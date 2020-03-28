from sklearn.ensemble import RandomForestClassifier

import numpy as np

import pandas as pd

np.random.seed(0)

from sklearn.metrics import accuracy_score,recall_score,confusion_matrix, classification_report



from sklearn.model_selection import train_test_split




#####  CLEANING STAGE ###############

df = pd.read_csv('datasets/recidivism_data_Analysis.csv')




df = df.drop(['Person_ID', 'MiddleName', 'DateOfBirth', 'Screening_Date', 'IsCompleted', 
	'IsDeleted', 'AssessmentReason', 'LastName', 'FirstName','RawScore'], axis = 1)


#### TO DROP THE BLANK VALUES IN THE SCORE TEXT FIELD - WE DROPPED COZ ONLY 2 SUCH TUPLES, HENCE WOULDNT MAKE MUCH OF DIFFERENCE. ####################
df['ScoreText'].replace('', np.nan, inplace=True)
df.dropna(subset=['ScoreText'], inplace=True)



# print(df.head(10))
df.to_csv('cleaned_data.csv')
# print(df.describe())


############### cleaned_data.csv is the final dataset we obtain #######################


df2 = pd.read_csv('cleaned_data.csv')



# df2['is_train'] = np.random.uniform(0,1, len(df2)) <= 0.75


###### TO CONVERT STRING TYPE ATTRIBUTES INTO NUMBERS ##############


dict1 = {'Male':0, 'Female':1}
df2['Sex_Code_Text'] = df2['Sex_Code_Text'].map(dict1)


dict1 = {'Broward County':1, 'DRRD':2, 'PRETRIAL': 3, 'Probation': 4}
df2['Agency_Text'] = df2['Agency_Text'].map(dict1)



dict1 = {'African-Am':1, 'African-American':2, 'Arabic':3, 'Asian':4, 'Caucasian':5,
 'Hispanic':6, 'Native American':7, 'Oriental':8, 'Other':9}
df2['Ethnic_Code_Text'] = df2['Ethnic_Code_Text'].map(dict1)


dict1 = {'All Scales':1, 'Risk and Prescreen':2}
df2['ScaleSet'] = df2['ScaleSet'].map(dict1)

dict1 = {'English':1, 'Spanish':2}
df2['Language'] = df2['Language'].map(dict1)


dict1 = {'Conditional Release':1, 'Deferred Sentencing':2, 'Other': 3, 'Parole Violator': 4, 'Post Sentence':5, 'Pretrial':6, 'Probation Violator':7}
df2['LegalStatus'] = df2['LegalStatus'].map(dict1)



dict1 = {'Jail Inmate':1, 'Parole':2, 'Pretrial Defendant':3, 'Prison Inmate':4, 'Probation':5,
 'Residential Program':6}
df2['CustodyStatus'] = df2['CustodyStatus'].map(dict1)


dict1 = {'Single':1, 'Significant Other':2, 'Unknown':3, 'Widowed':4, 'Divorced':5, 'Married':6, 'Separated':7}
df2['MaritalStatus'] = df2['MaritalStatus'].map(dict1)






dict1 = {'High':1, 'Medium':2, 'Low':3, 'Medium with Override Consideration':4}
df2['RecSupervisionLevelText'] = df2['RecSupervisionLevelText'].map(dict1)


dict1 = {'Risk of Failure to Appear':1, 'Risk of Recidivism':2, 'Risk of Violence':3}
df2['DisplayText'] = df2['DisplayText'].map(dict1)



dict1 = {'New':1, 'Copy':2}
df2['AssessmentType'] = df2['AssessmentType'].map(dict1)




dict1 = {'Low':1, 'Medium':2, 'High':3}
df2['ScoreText'] = df2['ScoreText'].map(dict1)


# print(df2.head(20))


######################################











########## SPLITTING AND APPLICATION OF ALGORITHM ########################
# train, test = df2[df2['is_train'] == True] , df2[df2['is_train'] == False]



features = df2.columns[1:17]

# print(features)

X_1 = np.array(df2[features])

y_1 = np.array(df2['ScoreText'])

X_train, X_test, y_train, y_test = train_test_split(X_1, y_1, test_size=0.30, random_state=42)


print("no of training samples", len(y_train))  ## 45678
print("no of testing samples", len(y_test)) ## 15120



from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)



from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train, y_train)

# print("no of training samples", len(train))  ## 45678
# print("no of testing samples", len(test)) ## 15120


# features = df2.columns[:15]

# print(features)
# y = train['ScoreText']



# print(y)
y_pred = classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))


print("accuracy_score - ",accuracy_score(y_test, y_pred)*100)


