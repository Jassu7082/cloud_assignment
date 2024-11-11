from flask import Flask, request, render_template
from flask_cors import CORS
import torch
import joblib
import re
from string import punctuation

app = Flask(__name__)
CORS(app)

# Load model and vectorizer
model_path = 'model_epoch_10.pth'
vectorizer_path = 'tfidf_vectorizer.pkl'
label_encoder_path = 'label_encoder.pkl'

# Define model class
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

# Inspect vectorizer
with open(vectorizer_path, 'rb') as f:
    vectorizer = joblib.load(f)

input_dim = vectorizer.vocabulary_.__len__()
print(f"Number of features in vectorizer: {input_dim}")

# Inspect label encoder
with open(label_encoder_path, 'rb') as f:
    label_encoder = joblib.load(f)

output_dim = len(label_encoder.classes_)
print(f"Number of classes: {output_dim}")

# Initialize and load model with correct dimensions
model = SimpleNN(input_dim, output_dim).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Define stopwords manually
STOPWORDS = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 
    'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 
    'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 
    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 
    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 
    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 
    'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 
    'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 
    'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 
    'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 
    'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn'
])

# Preprocessing function using regular expressions and manual stopword removal
def preprocess_text(text):
    # Remove HTML tags
    text = re.sub('<.*?>', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove special characters and punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove mentions (@username)
    text = re.sub(r'@\S+', '', text)
    # Remove hashtags (#hashtag)
    text = re.sub(r'#\S+', '', text)
    
    # Tokenize the text by splitting it into words
    tokens = text.split()
    
    # Remove stopwords and lemmatize (simple rule-based lemmatization)
    processed_words = []
    for word in tokens:
        # Remove stopwords
        if word.lower() not in STOPWORDS:
            # Simple rule-based lemmatization (just removing 's' at the end for plurals)
            if word.endswith('s'):
                word = word[:-1]
            processed_words.append(word.lower())
    
    # Join the words back into a single string
    text = ' '.join(processed_words)
    return text

# Route to render index.html
@app.route('/')
def index():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
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

    return sentiment

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
