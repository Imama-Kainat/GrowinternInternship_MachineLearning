import numpy as np                    # NumPy is used for numerical operations
import matplotlib.pyplot as plt       #Matplotlib for plotting
import pandas as pd                   #Pandas for data manipulation
from sklearn.model_selection import train_test_split # scikit-learn for machine learning.
from sklearn.tree import DecisionTreeClassifier

# Read data from the CSV file
data = pd.read_csv(r"C:\Users\Mamai Nataki\Desktop\techno_hack_internship\machine_learning\train.csv")

# Extract features and labels
features = data.iloc[:, 1:].values  # assuming features are in columns 1 and onwards
labels = data.iloc[:, 0].values  # assuming labels are in the first column

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
# 80% of the data is used for training (X_train and y_train),
# and 20% is reserved for testing (X_test and y_test). The random_state parameter ensures reproducibility.
# Train Decision Tree classifier
clf = DecisionTreeClassifier()#initializes a Decision Tree classifier (clf) and trains it using the training data.
clf.fit(X_train, y_train)

# Predict labels for test data
predictions = clf.predict(X_test)
#The trained model (clf) is used to predict labels for the test set (X_test).
# Calculate accuracy
accuracy = np.sum(predictions == y_test) / len(y_test) * 100

# Print accuracy
print("Accuracy:", accuracy, "%")

# Optional: visualize features and decision boundaries
# The code uses a simple Decision Tree classifier for classification.
# It's a basic example of a supervised learning task where the model is trained to predict the label of each data point based on its features.
# The accuracy indicates how well the model performs on unseen data.
# Visualize some images with their predicted labels
num_images_to_visualize = 5
for i in range(num_images_to_visualize):
    # Display the image
    img = X_test[i].reshape(28, 28)  # assuming images are 8x8 pixels
    plt.imshow(img, cmap='gray')

    # Print the true and predicted labels
    true_label = y_test[i]
    predicted_label = predictions[i]
    plt.title(f'True: {true_label}, Predicted: {predicted_label}')

    plt.show()