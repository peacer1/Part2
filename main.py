import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from models.knn_model import train_knn
from models.decision_tree_model import train_decision_tree

#1.Φορτώνουμε τα δεδομένα
df = pd.read_csv("data/loan_data.csv")

#2.Δείγμα των πρώτων γραμμών
print("Πρώτες 5 γραμμές:")
print(df.head())

#3.Έλεγχος για κενές τιμές
print("\nΚενές τιμές:")
print(df.isnull().sum())

#4.Κατανομή της στήλης loan_approved
print("\nΚατανομή εγκρίσεων:")
print(df["loan_approved"].value_counts())

#5.Μετατροπή κατηγορικής μεταβλητής employment_status
le = LabelEncoder()
df["employment_status"] = le.fit_transform(df["employment_status"])

#6.Ορισμός X (features) και y (στόχος)
X = df.drop("loan_approved", axis=1)
y = df["loan_approved"]

print("\nX στήλες:")
print(X.columns)

print("\nΣχήμα X και y:")
print(f"X: {X.shape}, y: {y.shape}")

# 7. Διαχωρισμός σε training και test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print("\nTrain/Test Split:")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Εκπαίδευση και αξιολόγηση
knn = train_knn(X_train, y_train, X_test, y_test)
tree = train_decision_tree(X_train, y_train, X_test, y_test)