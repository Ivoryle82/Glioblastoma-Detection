import os
import pandas as pd
import SimpleITK as sitk
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Function to extract features from MRI images
def extract_features(image_paths, mask_paths):
    features = []
    for img_path, mask_path in zip(image_paths, mask_paths):
        try:
            # Read image and mask
            image = sitk.ReadImage(img_path)
            mask = sitk.ReadImage(mask_path)

            # Example: Compute simple intensity-based features (mean, variance)
            statistics_filter = sitk.StatisticsImageFilter()
            statistics_filter.Execute(image, mask)
            mean_intensity = statistics_filter.GetMean()
            variance_intensity = statistics_filter.GetVariance()

            # Example: Append computed features to the list
            features.append([mean_intensity, variance_intensity])
        except Exception as e:
            print(f"Error extracting features for {img_path}: {e}")
    return features

# Load data from CSV file
def load_data(csv_file):
    data = pd.read_csv(csv_file)
    return data['Image'].tolist(), data['Mask'].tolist(), data['Label'].tolist()

# Main function for training and evaluation
def main():
    # Paths
    train_csv_file = 'train_data.csv'
    test_csv_file = 'test_data.csv'

    # Load training data
    train_image_paths, train_mask_paths, train_labels = load_data(train_csv_file)

    # Extract features from training images
    train_features = extract_features(train_image_paths, train_mask_paths)

    # Train a classifier (Example: Support Vector Machine)
    classifier = SVC(kernel='linear')
    classifier.fit(train_features, train_labels)

    # Load testing data
    test_image_paths, test_mask_paths, test_labels = load_data(test_csv_file)

    # Extract features from testing images
    test_features = extract_features(test_image_paths, test_mask_paths)

    # Predict labels for testing data
    predicted_labels = classifier.predict(test_features)

    # Evaluate the model
    accuracy = accuracy_score(test_labels, predicted_labels)
    print(f"Accuracy: {accuracy}")

if __name__ == "__main__":
    main()
