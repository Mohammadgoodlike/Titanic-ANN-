import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from scipy.stats import chi2_contingency
from tensorflow import keras
from tensorflow.keras.layers import Dense

import matplotlib.pyplot as plt

train_data = pd.read_csv('data_frame/train.csv')
test_data = pd.read_csv('data_frame/test.csv')
real_data = pd.read_csv('data_frame/statuslife.csv')

train_data['Age'] = train_data['Age'].fillna(train_data['Age'].mean())

test_data['Age'] = test_data['Age'].fillna(test_data['Age'].mean())

train_data['Cabin'] = train_data['Cabin'].fillna(train_data['Cabin'].mode()[0])  # Use mode for categorical
test_data['Cabin'] = test_data['Cabin'].fillna(test_data['Cabin'].mode()[0])

label_encoder = LabelEncoder()

train_data['Sex'] = label_encoder.fit_transform(train_data['Sex'])
test_data['Sex'] = label_encoder.fit_transform(test_data['Sex'])
train_data['Ticket'] = label_encoder.fit_transform(train_data['Ticket'])
test_data['Ticket'] = label_encoder.fit_transform(test_data['Ticket'])
train_data['Cabin'] = label_encoder.fit_transform(train_data['Cabin'])
test_data['Cabin'] = label_encoder.fit_transform(test_data['Cabin'])

# Feature selection
# PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked

X_train = train_data[['Sex', 'Age', 'Fare', 'Ticket', 'Cabin']]
y_train = train_data['Survived']
X_test = test_data[['Sex', 'Age', 'Fare', 'Ticket', 'Cabin']]
y_test = real_data['Survived']

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = keras.Sequential()

model.add(Dense(units=128, activation='relu', input_shape=(np.shape(X_train)[1],)))

model.add(Dense(units=128, activation='relu'))

model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=70, batch_size=32, validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test)
print(f"\n\t Model Accuracy: {accuracy * 100:.2f}%  \t\t Loss : {loss * 100 :.2f}%")

y_pred = (model.predict(X_test) > 0.5).astype("int32")

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')

# plt.savefig('confusion_matrix_plot.png')
plt.show()

features = ['Sex', 'Age', 'Fare', 'Ticket', 'Cabin']

train_data['Combined'] = train_data[features].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)

test_data['Combined'] = test_data[features].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)

train_combined = train_data['Combined'].value_counts().sort_index()
test_combined = test_data['Combined'].value_counts().sort_index()

contingency_table_combined = pd.DataFrame([train_combined, test_combined], index=['Train', 'Test']).fillna(0)

chi2_combined, p_combined, dof_combined, expected_combined = chi2_contingency(contingency_table_combined)

print(f"Chi2: {chi2_combined:.4f}, p-value: {p_combined:.4f}")

import pandas as pd

df = pd.DataFrame(contingency_table_combined)
df.to_csv('matrix.csv')
