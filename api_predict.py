"""
Project: Stunting Prediction API
Author: Silvio Christian, Joe
Description:
    This module serves as the backend inference engine using Flask.
    It handles model loading, input preprocessing (Encoding/Scaling),
    and serving predictions via a RESTful API endpoint.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np 
import os 

# ==========================================
# 1. App Initialization
# ==========================================
app = Flask(__name__)
CORS(app) # Enable Cross-Origin Resource Sharing for frontend integration

# ==========================================
# 2. Artifact Loading (Model & Preprocessors)
# ==========================================
# We load the serialized model and preprocessors at startup to ensure efficient inference.
try:
    # 1. Main Classification Model
    model = joblib.load("best_model.joblib")
    
    # 2. Scaler for numerical features (Age, Height, Weight)
    scaler = joblib.load("scaler.joblib")
    
    # 3. Encoder for categorical feature 'Jenis Kelamin' (Input)
    # Ensure filename matches exactly
    jk_encoder = joblib.load("Jenis Kelamin_encoder.joblib")
    
    # 4. Encoder for target label 'Stunting' (Output)
    stunting_encoder = joblib.load("Stunting_encoder.joblib")
    
    print("System: All 4 artifacts (model, scaler, encoders) loaded successfully!")

except Exception as e:
    # Log critical errors if artifacts fail to load
    print(f"System Error: Failed to load models: {e}")
    model = None
    scaler = None
    jk_encoder = None
    stunting_encoder = None

# ==========================================
# 3. Prediction Endpoint
# ==========================================
@app.route('/predict', methods=['POST'])
def predict():    
    # Validation: Ensure all artifacts are loaded before processing
    if not all([model, scaler, jk_encoder, stunting_encoder]):
        return jsonify({"error": "Server Error: Model artifacts not initialized."}), 500

    try:
        # 1. Parse JSON payload from client (React/Frontend)
        data = request.get_json()

        # Expected JSON Structure:
        # {
        #    "jenis_kelamin": "Laki-laki",  // Categorical
        #    "umur": 24,                    // Numerical
        #    "tinggi": 85.5,                // Numerical
        #    "berat": 10.1                  // Numerical
        # }

        # --- PREPROCESSING PIPELINE ---
        # Critical: We must replicate the exact preprocessing steps used during training.

        # 2. Extract raw data
        jk_string = data['jenis_kelamin']
        umur = data['umur']
        tinggi = data['tinggi']
        berat = data['berat']

        # 3. Categorical Encoding (String -> Numerical)
        # Transform 'Jenis Kelamin' using the fitted LabelEncoder
        jk_encoded = jk_encoder.transform([jk_string])[0]

        # 4. Numerical Scaling (Standardization/Normalization)
        # Order must match training: ['Umur', 'Tinggi Badan', 'Berat Badan']
        numerical_features = [[umur, tinggi, berat]]
        scaled_features = scaler.transform(numerical_features)
        
        # Flatten the scaled result
        umur_scaled = scaled_features[0][0]
        tinggi_scaled = scaled_features[0][1]
        berat_scaled = scaled_features[0][2]

        # 5. Feature Assembly
        # Combine all processed features in the correct order expected by the model
        final_features_list = [jk_encoded, umur_scaled, tinggi_scaled, berat_scaled]
        
        # 6. Reshape for Inference
        # Convert to 2D Numpy Array (1 sample, n features)
        final_features = [np.array(final_features_list)]
        
        # --- INFERENCE ---

        # 7. Generate Prediction
        prediction_encoded = model.predict(final_features)
        
        # 8. Post-processing (Numerical -> String Label)
        # Convert the predicted class index back to a readable label (e.g., "Stunted")
        prediction_string = stunting_encoder.inverse_transform(prediction_encoded)

        # 9. Extract result string
        output = prediction_string[0]

        # 10. Return success response
        return jsonify({'prediction': output})

    except KeyError as e:
        # Handle missing keys in JSON payload
        return jsonify({"error": f"Bad Request: Missing JSON key {str(e)}. Required: 'jenis_kelamin', 'umur', 'tinggi', 'berat'."}), 400
    
    except Exception as e:
        # Handle unexpected errors during processing
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 400

# ==========================================
# 4. Server Execution
# ==========================================
if __name__ == '__main__':
    # Fetch PORT from environment variables (required for Deployment platforms like Heroku)
    port = int(os.environ.get('PORT', 5000))
    
    # Run Flask app
    app.run(debug=True, port=port)
