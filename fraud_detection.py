# write your code here
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Data preparation
data = pd.read_csv("creditcard.csv").drop(columns=["Time"])

# print(data.iloc[:20])

features = data.drop(columns=["Class"])
target = pd.DataFrame(data["Class"])

# Train test split
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=1)

# Scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
y_train = scaler.fit_transform(y_train)
y_test = scaler.transform(y_test)

y_train = y_train.reshape(len(y_train), )
y_train = y_train.astype('int')

y_test = y_test.reshape(len(y_test), )
y_test = y_test.astype('int')
# print(x_train[:2])

# Model implementing
model = LogisticRegression(random_state=2)
model.fit(x_train, y_train)
# print(model.get_params())

# Model evaluation
predictions = model.predict(x_test)

print(model.score(x_train, y_train))
print(predictions)
print(confusion_matrix(y_test, predictions))
