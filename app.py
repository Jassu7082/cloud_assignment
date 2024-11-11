from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from string import punctuation

app = Flask(__name__)
CORS(app)

# Load model and vectorizer paths
model_path = 'model_epoch_10.pth'
vectorizer_path = 'tfidf_vectorizer.pkl'
label_encoder_path = 'label_encoder.pkl'

# Define the model class
class SimpleNN(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 128)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Initialize and load model
device = torch.device("cpu")

# Load the vectorizer
with open(vectorizer_path, 'rb') as f:
    vectorizer = joblib.load(f)

input_dim = vectorizer.vocabulary_.__len__()
print(f"Number of features in vectorizer: {input_dim}")

# Load the label encoder
with open(label_encoder_path, 'rb') as f:
    label_encoder = joblib.load(f)

output_dim = len(label_encoder.classes_)
print(f"Number of classes: {output_dim}")

# Initialize and load model with correct dimensions
model = SimpleNN(input_dim, output_dim).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Initialize NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# Ensure WordNet is loaded
from nltk.corpus import wordnet as wn
wn.ensure_loaded()

# Preprocessing function
lemma = WordNetLemmatizer()

def preprocess_text(text):
    text = re.sub('<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\S+', '', text)  # Remove mentions
    text = re.sub(r'#\S+', '', text)  # Remove hashtags

    tokens = nltk.word_tokenize(text)
    punc = list(punctuation)
    words = [word for word in tokens if word not in punc]
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.lower() not in stop_words]
    words = [lemma.lemmatize(word) for word in words]

    return ' '.join(words)

# Route to render index.html
@app.route('/')
def index():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'tweet' not in data:
            return jsonify({'error': 'Tweet is missing'}), 400
        
        tweet = data['tweet']
        
        # Preprocess input text
        processed_tweet = preprocess_text(tweet)
        
        # Vectorize preprocessed text
        vectorized_tweet = vectorizer.transform([processed_tweet]).toarray()
        vectorized_tweet_tensor = torch.tensor(vectorized_tweet, dtype=torch.float32).to(device)
        
        # Predict sentiment
        with torch.no_grad():
            outputs = model(vectorized_tweet_tensor)
            prediction = torch.argmax(outputs, dim=1).cpu().numpy()[0]
            sentiment = label_encoder.inverse_transform([prediction])[0]

        return jsonify({'sentiment': sentiment})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
