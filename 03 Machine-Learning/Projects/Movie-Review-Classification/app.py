import pickle
import re
from flask import Flask, jsonify, render_template, request
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

nltk.download('stopwords')
nltk.download('wordnet')

# Load ONLY the pipeline model (which includes the vectorizer)
model = pickle.load(open('model.pkl', 'rb'))

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def preprocess_text(text):
  if not isinstance(text, str):
    return ''
  text = re.sub(r'[^a-zA-Z]', ' ', text).lower()
  tokens = text.split()
  cleaned = [
      lemmatizer.lemmatize(word) for word in tokens if word not in stop_words
  ]
  return ' '.join(cleaned)


@app.route('/')
def home():
  return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_review():
  try:
    data = request.get_json()
    raw_review = data.get('review', '')

    if not raw_review.strip():
      return jsonify({'error': 'Please enter your review to classify'}), 400

    # 1. Preprocess text
    cleaned_review = preprocess_text(raw_review)

    # 2. Predict directly using the pipeline (it handles TF-IDF transformation automatically)
    prediction_result = model.predict([cleaned_review])[0]

    return jsonify({'prediction': prediction_result})

  except Exception as e:
    return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
  app.run(debug=True, port=5000)