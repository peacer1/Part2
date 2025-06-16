from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report


def train_decision_tree(X_train, y_train, X_test, y_test):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\nΑποτελέσματα Decision Tree:")
    print(classification_report(y_test, predictions))

    print("\nΣημαντικότητα χαρακτηριστικών:")
    for name, importance in zip(X_train.columns, model.feature_importances_):
        print(f"{name}: {importance:.4f}")

    return model
