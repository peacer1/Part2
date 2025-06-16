from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

def train_knn(X_train, y_train, X_test, y_test):
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\nΑποτελέσματα KNN:")
    print(classification_report(y_test, predictions))

    return model
