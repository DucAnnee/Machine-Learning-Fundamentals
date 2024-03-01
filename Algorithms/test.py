import pandas as pd
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("iris.csv")

# Visualize class distribution
class_counts = Counter(data["class"])
plt.bar(class_counts.keys(), class_counts.values())
plt.xlabel("Class")
plt.ylabel("Number of data points")
plt.title("Distribution of classes in the dataset")
plt.show()

# Separate features and target
X = data.drop("class", axis=1)
y = data["class"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create and train KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_scaled, y)

# New data point (replace with your actual data)
new_data = [[1.5, 5.5, 8.5, 11.5]]

# Scale new data and predict
new_data_scaled = scaler.transform(new_data)
predicted_class = knn.predict(new_data_scaled)

# Print the predicted class
print("Predicted class:", predicted_class[0])
