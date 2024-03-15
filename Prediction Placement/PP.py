import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

train_df = pd.read_excel('C:/Users/Mazaher/Desktop/05 Input Data/Prediction of Placement Status Data/01 Train Data.xlsx')
test_df = pd.read_excel('C:/Users/Mazaher/Desktop/05 Input Data/Prediction of Placement Status Data/02 Test Data.xlsx')

print(train_df.dtypes)

# Convert all columns to strings
train_df = train_df.astype(str)
test_df = test_df.astype(str)
combined_df = pd.concat([train_df, test_df], axis=0)

le = LabelEncoder()

for column in combined_df.columns:
    if combined_df[column].dtype == 'object':
        combined_df[column] = le.fit_transform(combined_df[column])

train_df = combined_df.iloc[:len(train_df)]
test_df = combined_df.iloc[len(train_df):]

X_train = train_df.drop('Placement Status', axis=1)
y_train = train_df['Placement Status']
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print('Accuracy:', accuracy_score(y_test, predictions))

unseen_labels = set(test_df['Placement Status']) - set(le.classes_)
if unseen_labels:
    print("Instances with previously unseen labels found in test data. Dropping them.")
    test_df = test_df[~test_df['Placement Status'].isin(unseen_labels)]

if not test_df.empty:
    for column in test_df.columns:
        if test_df[column].dtype != 'float64' and test_df[column].dtype != 'int64':
            test_df[column] = test_df[column].astype(str)

    for column in test_df.columns:
        if test_df[column].dtype == 'object':
            test_df[column] = le.transform(test_df[column])

    predictions_test_data = model.predict(test_df.drop('Placement Status', axis=1))

    if 'Placement Status' in test_df.columns:
        accuracy_test_data = accuracy_score(test_df['Placement Status'], predictions_test_data)
        print('Accuracy on test data:', accuracy_test_data)
    else:
        print('Predictions on test data:', predictions_test_data)
else:
    print("No instances available for testing after filtering out unseen labels.")
