import os
import pickle
import pandas as pd
from flask import Flask, request, render_template, session, send_file
from flask_cors import cross_origin

# Load the model
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

# Flask app configuration
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for session handling

# Path to the history CSV file
history_csv_path = os.path.join(os.path.dirname(__file__), "history.csv")

# Function to load history from CSV
def load_history():
    if os.path.exists(history_csv_path):
        return pd.read_csv(history_csv_path).to_dict('records')
    return []

# Function to save history to CSV
def save_history(history):
    pd.DataFrame(history).to_csv(history_csv_path, index=False)

# Route to download search history as CSV
@app.route('/download_history', methods=["GET"])
def download_history():
    if os.path.exists(history_csv_path):
        return send_file(history_csv_path, as_attachment=True)
    return "No history available to download."

# Home Page
@app.route('/')
def home():
    # Set festival offers in the session
    session['festival_offers'] = [
        "10% off on all flights during Diwali!",
        "Christmas special: Buy one get one free!",
        "New Year Bonanza: Flat 20% off on all international flights!"
    ]
    return render_template('home.html')

# Form Page with History
@app.route('/form')
def form():
    # Retrieve search history from the CSV file
    history = load_history()
    return render_template('form.html', history=history)

# Prediction Result Page
@app.route("/predict", methods=["GET", "POST"])
@cross_origin()
def predict():
    if request.method == "POST":
        try:
            dep_date = request.form.get('Dep_Time')
            if dep_date:
                dep_date = pd.to_datetime(dep_date, format='%Y-%m-%dT%H:%M')
                Journey_day = dep_date.day
                Journey_month = dep_date.month
                dep_hour = dep_date.hour
                dep_min = dep_date.minute
            else:
                raise ValueError("Departure date is missing")

            arrival_date = request.form.get('Arrival_Time')
            if arrival_date:
                arrival_date = pd.to_datetime(arrival_date, format='%Y-%m-%dT%H:%M')
                arrival_hour = arrival_date.hour
                arrival_min = arrival_date.minute
            else:
                raise ValueError("Arrival date is missing")

            dur_hour = abs(arrival_hour - dep_hour)
            dur_min = abs(arrival_min - dep_min)
            Total_stops = int(request.form["stops"])
            airline = request.form.get('airline')

            # One-hot encoding for the airline field
            airlines = ['Jet Airways', 'IndiGo', 'Air India', 'Multiple carriers', 'SpiceJet', 'Vistara',
                        'GoAir', 'Multiple carriers Premium economy', 'Jet Airways Business', 'Vistara Premium economy',
                        'Trujet']
            airline_data = {f"{airline.replace(' ', '_')}": 0 for airline in airlines}
            if airline in airlines:
                airline_data[airline.replace(' ', '_')] = 1

            Source = request.form["Source"]
            Destination = request.form["Destination"]

            # One-hot encoding for the source and destination fields
            sources = ['Delhi', 'Kolkata', 'Mumbai', 'Chennai']
            destinations = ['Cochin', 'Delhi', 'New_Delhi', 'Hyderabad', 'Kolkata']

            source_data = {f"s_{source}": 0 for source in sources}
            if Source in sources:
                source_data[f"s_{Source}"] = 1

            destination_data = {f"d_{Destination.replace(' ', '_')}": 0 for destination in destinations}
            if Destination in destinations:
                destination_data[f"d_{Destination.replace(' ', '_')}"] = 1

            # Prepare data for prediction
            data = [Total_stops, Journey_day, Journey_month, dep_hour, dep_min, arrival_hour, arrival_min, dur_hour, dur_min]
            data.extend(list(airline_data.values()))
            data.extend(list(source_data.values()))
            data.extend(list(destination_data.values()))

            # Ensure the length of data matches the expected features
            expected_features = 29
            if len(data) < expected_features:
                data.extend([0] * (expected_features - len(data)))  # Add missing features with default values

            if len(data) == expected_features:
                prediction = model.predict([data])
                output = round(prediction[0], 2)

                # Save current input into the CSV history
                history = load_history()
                history.append({
                    'airline': airline,
                    'source': Source,
                    'destination': Destination,
                    'stops': Total_stops,
                    'predicted_price': output
                })
                save_history(history)  # Update CSV with new history

                # Predict last month price
                last_month_date = dep_date - pd.DateOffset(days=30)
                last_month_data = data.copy()
                last_month_data[1] = last_month_date.day
                last_month_data[2] = last_month_date.month
                last_month_prediction = model.predict([last_month_data])
                last_month_price = round(last_month_prediction[0], 2)

                # Predict current price
                current_data = data.copy()
                current_price_prediction = model.predict([current_data])
                current_price = round(current_price_prediction[0], 2)

                return render_template('result.html', prediction_text=f"Your Flight price is Rs. {output}", last_month_price=last_month_price, current_price=current_price, festival_offers=session.get('festival_offers', []), dep_date=dep_date, arrival_date=arrival_date, Total_stops=Total_stops, airline=airline, Source=Source, Destination=Destination)
            else:
                return render_template('result.html', prediction_text="Error: Feature length mismatch. Please check the input data.", festival_offers=session.get('festival_offers', []))
        except Exception as e:
            return render_template('result.html', prediction_text=f"Error: {str(e)}", festival_offers=session.get('festival_offers', []))

    return render_template('form.html')

# New route for predicting prices for last month, current date, and future dates
@app.route("/predict_dates", methods=["POST"])
@cross_origin()
def predict_dates():
    try:
        dep_date = request.form.get('Dep_Time')
        Journey_day = int(pd.to_datetime(dep_date, format='%Y-%m-%dT%H:%M').day)
        Journey_month = int(pd.to_datetime(dep_date, format='%Y-%m-%dT%H:%M').month)
        dep_hour = int(pd.to_datetime(dep_date, format='%Y-%m-%dT%H:%M').hour)
        dep_min = int(pd.to_datetime(dep_date, format='%Y-%m-%dT%H:%M').minute)
        arrival_date = request.form.get('Arrival_Time')
        arrival_hour = int(pd.to_datetime(arrival_date, format='%Y-%m-%dT%H:%M').hour)
        arrival_min = int(pd.to_datetime(arrival_date, format='%Y-%m-%dT%H:%M').minute)
        dur_hour = abs(arrival_hour - dep_hour)
        dur_min = abs(arrival_min - dep_min)
        Total_stops = int(request.form["stops"])
        airline = request.form.get('airline')

        # One-hot encoding for the airline field
        airlines = ['Jet Airways', 'IndiGo', 'Air India', 'Multiple carriers', 'SpiceJet', 'Vistara',
                    'GoAir', 'Multiple carriers Premium economy', 'Jet Airways Business', 'Vistara Premium economy',
                    'Trujet']
        airline_data = {f"{airline.replace(' ', '_')}": 0 for airline in airlines}
        if airline in airlines:
            airline_data[airline.replace(' ', '_')] = 1

        Source = request.form["Source"]
        Destination = request.form["Destination"]

        # One-hot encoding for the source and destination fields
        sources = ['Delhi', 'Kolkata', 'Mumbai', 'Chennai']
        destinations = ['Cochin', 'Delhi', 'New_Delhi', 'Hyderabad', 'Kolkata']

        source_data = {f"s_{source}": 0 for source in sources}
        if Source in sources:
            source_data[f"s_{Source}"] = 1

        destination_data = {f"d_{Destination.replace(' ', '_')}": 0 for destination in destinations}
        if Destination in destinations:
            destination_data[f"d_{Destination.replace(' ', '_')}"] = 1

        # Prepare data for prediction
        data = [Total_stops, Journey_day, Journey_month, dep_hour, dep_min, arrival_hour, arrival_min, dur_hour, dur_min]
        data.extend(list(airline_data.values()))
        data.extend(list(source_data.values()))
        data.extend(list(destination_data.values()))

        # Ensure the length of data matches the expected features
        expected_features = 29
        if len(data) < expected_features:
            data.extend([0] * (expected_features - len(data)))  # Add missing features with default values

        if len(data) == expected_features:
            predictions = {}
            for offset in [-30, 0, 30]:  # Last month, current date, next month
                future_date = pd.to_datetime(dep_date, format='%Y-%m-%dT%H:%M') + pd.DateOffset(days=offset)
                data[1] = future_date.day
                data[2] = future_date.month
                prediction = model.predict([data])
                predictions[offset] = round(prediction[0], 2)

            return render_template('result.html', prediction_text="Predictions for multiple dates", predictions=predictions, festival_offers=session.get('festival_offers', []))
        else:
            return render_template('result.html', prediction_text="Error: Feature length mismatch. Please check the input data.", festival_offers=session.get('festival_offers', []))
    except Exception as e:
        return render_template('result.html', prediction_text=f"Error: {str(e)}", festival_offers=session.get('festival_offers', []))

# Route to clear search history
@app.route("/clear_history", methods=["POST"])
def clear_history():
    if os.path.exists(history_csv_path):
        os.remove(history_csv_path)  # Remove history CSV file
    return render_template('form.html', history=[])

if __name__ == "__main__":
    app.run(debug=True)
