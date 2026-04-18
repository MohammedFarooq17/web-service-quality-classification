import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("data.csv")

le = LabelEncoder()
df['quality'] = le.fit_transform(df['quality'])


X = df[['response_time', 'uptime', 'latency']]
y = df['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)

sample = [[150, 98, 40]]
prediction = model.predict(sample)
print("Predicted Quality:", le.inverse_transform(prediction))