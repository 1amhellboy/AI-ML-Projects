from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Load dataset
iris = load_iris()

# 2. Split into training & testing
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2
)

# 3. Choose algorithm
model = LogisticRegression(max_iter=200)

# 4. Train the model
model.fit(X_train, y_train)

# 5. Make predictions
preds = model.predict(X_test)

# 6. Evaluate
print("Accuracy:", accuracy_score(y_test, preds))
