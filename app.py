from flask import Flask, request, jsonify
import pickle
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
with open('bus_maintenance_data.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
    print(type(model))  # Check the type of the model

@app.route('/')
def home():
    return "Bus Maintenance Prediction API"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the request
        data = request.json
        
        # Convert data into a DataFrame for prediction
        input_data = pd.DataFrame([data])
        print(input_data)  # Debugging: Print the input data

        # Ensure the input data matches the model's expected feature order
        # input_data = input_data[['Mileage', 'Bus_Age', 'Trips_per_Day', 'Avg_Speed', 'Fuel_Type', 'Road_Type', 'Stops']]
        
        # Make predictions using the loaded model
        prediction = model.predict(input_data)
        
        # Return the prediction as a JSON response
        return jsonify({'maintenance_cost_prediction': prediction[0]})
    
    except Exception as e:
        print(f"Error: {str(e)}")  # Print the error to the console
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
