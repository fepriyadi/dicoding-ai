from flask import Flask, request, jsonify
from sklearn.preprocessing import LabelEncoder
import pandas as pd

import joblib

app = Flask(__name__)
joblib_model = joblib.load('/Users/fepriyadi/Documents/ML/gbr_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    label_encoder = LabelEncoder()
    data = request.json['data']

    # Apply the same label encoding
    df = pd.DataFrame(data, columns=["video_id", "title", "channellId", "channelTitle","categoryId", "trending_date", "tags",
    "view_count","likes", "dislikes", "comment_count","thumbnail_link","description"])

    df_lencoder = pd.DataFrame(data)
    categorical_features = df.select_dtypes(include=['object']).columns

    for col in categorical_features:
        df_lencoder[col] = label_encoder.transform(df[col])

    prediction = joblib_model.predict(df_lencoder)
    return jsonify({'prediction': prediction.tolist()})
if __name__ == '__main__':
    app.run(debug=True)