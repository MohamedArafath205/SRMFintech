import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU, LSTM, Dropout

def load_data():
    data = pd.read_csv("fintech3.csv")
    return data

def preprocess_data(data):
    data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
    data['Insurance'] = data['Insurance'].map({'Yes': 1, 'No': 0})
    data['Demographic'] = data['Demographic'].map({'Rural': 0, 'Suburban': 1, 'Urban': 2})
    data['Marital_status'] = data['Marital_status'].map({'Married': 1, 'Single': 0})
    data['Properties'] = data['Properties'].map({'Apartment': 1, 'Condo': 2, 'House': 3})
    data['Emp_status'] = data['Emp_status'].map({'Self Employed': 1, 'Employee': 2, 'Entrepreneur': 3})
    data['Properties'].fillna(0, inplace=True)
    return data

def main():
    st.title("SRM FinTech Analysis")

    data = load_data()
    st.write("Data Shape:", data.shape)
    st.write("Data Overview:")
    st.write(data.head())

    data = preprocess_data(data)
    st.write("Data after Preprocessing:")
    st.write(data.head())

    X = data.drop("Fin_Cat", axis=1)
    y = data["Fin_Cat"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Gaussian Naive Bayes": GaussianNB(),
        "Decision Tree Classifier": DecisionTreeClassifier(random_state=42),
        "Random Forest Classifier": RandomForestClassifier(random_state=42)
    }

    for model_name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"{model_name} Accuracy:", accuracy)

    rf_classifier = models["Random Forest Classifier"]
    feature_importances = rf_classifier.feature_importances_
    feature_names = X.columns
    indices = np.argsort(feature_importances)[::-1]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(range(X.shape[1]), feature_importances[indices], align="center")
    ax.set_xticks(range(X.shape[1]))
    ax.set_xticklabels(feature_names[indices], rotation=45)
    ax.set_title("Feature Importances - Random Forest")
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(12, 8))
    plot_tree(rf_classifier.estimators_[0], feature_names=X.columns, filled=True, rounded=True, ax=ax)
    ax.set_title("Random Forest Visualization")
    st.pyplot(fig)

    model = Sequential()
    model.add(LSTM(units=64, input_shape=(1, X_train_scaled.shape[1]), return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(units=32))
    model.add(Dropout(0.5))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
    X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

    model.fit(X_train_reshaped, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)

    loss, accuracy = model.evaluate(X_test_reshaped, y_test)
    st.write("Test Loss (RNN):", loss)
    st.write("Test Accuracy (RNN):", accuracy)

    input_data = np.array([38, 1, 68000, 14000, 9500, 3000, 3, 1, 730, 2, 4, 1, 3, 3]).reshape(1, 1, -1)
    input_data_scaled = scaler.transform(input_data.reshape(1, -1))
    predicted_outcome_rnn = model.predict(input_data_scaled.reshape(1, 1, -1))
    st.write("Predicted Outcome (RNN):", predicted_outcome_rnn)

    input_weights = model.layers[0].get_weights()[0]
    feature_importances_rnn = np.sum(np.abs(input_weights), axis=1)
    feature_names_rnn = X.columns
    indices_rnn = np.argsort(feature_importances_rnn)[::-1]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(range(len(feature_importances_rnn)), feature_importances_rnn[indices_rnn], align="center")
    ax.set_xticks(range(len(feature_importances_rnn)))
    ax.set_xticklabels(feature_names_rnn[indices_rnn], rotation=45)
    ax.set_title("Feature Importances - RNN Model")
    st.pyplot(fig)

if __name__ == "__main__":
    main()
