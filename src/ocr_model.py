import pandas as pd
import pytesseract
import cv2
import os
from urllib.request import urlretrieve
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Set the path for Tesseract executable (Update it according to your system)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows path

# Function to download images from a URL
def download_image(image_url, filename):
    try:
        urlretrieve(image_url, filename)
    except Exception as e:
        print(f"Failed to download image {image_url}: {e}")

# Function to preprocess image (Optional: Can be tuned for better OCR accuracy)
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

# Function to extract text from an image using Tesseract
def extract_text_from_image(image_path):
    processed_image = preprocess_image(image_path)
    if processed_image is None:
        return ""
    text = pytesseract.image_to_string(processed_image)
    return text

# Function to prepare data for training
def prepare_data(training_file, sample_size=None):
    df = pd.read_csv(training_file)
    
    if sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    texts = []
    labels = []
    
    for idx, row in df.iterrows():
        image_url = row['image_link']
        entity_value = row['entity_value']
        image_path = f"temp_images/{idx}.jpg"  # Temp storage for the image
        
        download_image(image_url, image_path)
        text = extract_text_from_image(image_path)
        texts.append(text)
        labels.append(entity_value)
    
    return texts, labels

# Function to train the model
def train_model(training_file, sample_size=None):
    texts, labels = prepare_data(training_file, sample_size)
    
    # Encode labels
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    # Create a text processing and model pipeline
    model = make_pipeline(
        CountVectorizer(),  # Converts text to feature vectors
        LogisticRegression(max_iter=1000)  # Basic classification model with increased iterations
    )

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(texts, labels_encoded, test_size=0.2, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Print model accuracy
    print(f"Model accuracy: {model.score(X_test, y_test) * 100:.2f}%")

    return model, label_encoder

# Function to match text with allowed units (if applicable)
def match_units(entity_name, extracted_text):
    # For simplicity, we're not matching units in this example
    # Update as needed based on your specific use case
    return extracted_text

# Function to generate predictions
def generate_predictions(test_file, output_file, model=None, label_encoder=None, num_records=5):
    test_data = pd.read_csv(test_file)
    test_data = test_data.head(num_records)
    predictions = []

    for idx, row in test_data.iterrows():
        image_url = row['image_link']
        entity_name = row['entity_name']
        index = row['index']

        image_path = f"images/{index}.jpg"
        download_image(image_url, image_path)

        extracted_text = extract_text_from_image(image_path)

        if model and label_encoder:
            # Use the trained model to predict the entity value
            entity_value_prediction = model.predict([extracted_text])[0]
            entity_value = label_encoder.inverse_transform([entity_value_prediction])[0]
        else:
            entity_value = row['entity_value']  # Use the actual value if no model

        prediction = match_units(entity_name, extracted_text)

        predictions.append({"index": index, "prediction": prediction})

        print(f"Index: {index}, Prediction: {prediction}")

    output_df = pd.DataFrame(predictions)
    output_df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    # Define file paths
    training_file = os.path.abspath("D:/amazon ml challenge/project/dataset/train.csv")
    test_file = os.path.abspath("D:/amazon ml challenge/project/dataset/test.csv")
    output_file = os.path.abspath("D:/amazon ml challenge/project/dataset/test_out.csv")
    
    # Train the model with a subset of the data
    model, label_encoder = train_model(training_file, sample_size=10000)  # Use a smaller subset for initial training

    # Generate predictions
    generate_predictions(test_file, output_file, model=model, label_encoder=label_encoder, num_records=5)
