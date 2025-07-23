from flask import Flask, request, render_template
import numpy as np
import pickle
import os
import logging

# Initialize the Flask application
app = Flask(__name__)

# Configure logging for production
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-change-in-production')

# Disable debug mode in production
app.config['DEBUG'] = False

# Load the trained model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    """Renders the home page with the input form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction request."""
    try:
        # Get the feature values from the form
        # The order must match the training data columns
        features = [
            float(request.form['age']),
            int(request.form['anaemia']),
            float(request.form['creatinine_phosphokinase']),
            int(request.form['diabetes']),
            float(request.form['ejection_fraction']),
            int(request.form['high_blood_pressure']),
            float(request.form['platelets']),
            float(request.form['serum_creatinine']),
            float(request.form['serum_sodium']),
            int(request.form['sex']),
            int(request.form['smoking']),
            float(request.form['time'])
        ]
    except (ValueError, KeyError) as e:
        logger.error(f"Error processing form data: {e}")
        return render_template('index.html', error="Invalid input data. Please check your entries.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return render_template('index.html', error="An unexpected error occurred. Please try again.")

    try:
        # Convert features to a numpy array and reshape for a single prediction
        final_features = np.array(features).reshape(1, -1)

        # Scale the input features using the loaded scaler
        scaled_features = scaler.transform(final_features)

        # Make prediction
        prediction = model.predict(scaled_features)
        prediction_prob = model.predict_proba(scaled_features)
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return render_template('index.html', error="Error during prediction. Please try again.")

    # Determine the output message
    if prediction[0] == 1:
        # Calculate probability for the "likely to die" class
        probability = round(prediction_prob[0][1] * 100, 2)
        output_message = f"High Risk of a Heart Failure Event"
        output_prob = f"Probability: {probability}%"
        result_class = "high-risk"
    else:
        # Calculate probability for the "not likely to die" class
        probability = round(prediction_prob[0][0] * 100, 2)
        output_message = f"Low Risk of a Heart Failure Event"
        output_prob = f"Probability: {probability}%"
        result_class = "low-risk"

    # Render the result page with the prediction
    return render_template('result.html', prediction_text=output_message, prediction_prob=output_prob, result_class=result_class)

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return render_template('index.html', error="Page not found."), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {error}")
    return render_template('index.html', error="Internal server error. Please try again later."), 500

if __name__ == "__main__":
    # Use environment variable for debug mode
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(debug=debug_mode)