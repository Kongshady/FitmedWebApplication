from flask import Flask, render_template, request, jsonify  
import joblib  
import pandas as pd  

app = Flask(__name__)  

# Load the trained models and vectorizers  
model1_path = "models/disease_knn_model.pkl"  # First model path  
vectorizer1_path = "models/knn_vectorizer.pkl"  # First model label encoder  
model2_path = "models/disease_model.pkl"  # Second model path (already present)  
vectorizer2_path = "models/vectorizer.pkl"  # Second model label encoder (already present)  
model3_path = "models/naive_bayes_model.pkl"  # Third model path  
vectorizer3_path = "models/nb_vectorizer.pkl"  # Third model vectorizer  

# Load models  
model1 = joblib.load(model1_path)  
label_encoder1 = joblib.load(vectorizer1_path)  
model2 = joblib.load(model2_path)  
label_encoder2 = joblib.load(vectorizer2_path)  
model3 = joblib.load(model3_path)  
vectorizer3 = joblib.load(vectorizer3_path)  

# Load medications dictionary  
with open('models/medications_model.pkl', 'rb') as f:  
    medications_dict = joblib.load(f)   
medications_dict = {k.lower(): v for k, v in medications_dict.items()}  

# Pages Routes  
@app.route('/')  
def home():  
    return render_template('index.html')  

@app.route('/about')  
def about():  
    return render_template('about.html')  

@app.route('/solutions')  
def solutions():  
    return render_template('solutions.html')  

@app.route('/contact')  
def contact():  
    return render_template('contact.html')  

@app.route('/predict_disease', methods=['POST'])  
def predict_disease():  
    try:   
        # Extract form data  
        fever = request.form.get('Fever')  
        cough = request.form.get('Cough')  
        fatigue = request.form.get('Fatigue')  
        difficulty = request.form.get('Difficulty')  
        age = float(request.form.get('Age'))  
        gender = "Male" if request.form.get('Gender').lower() == "male" else "Female"  
        blood_pressure = request.form.get('BloodPressure')  
        cholesterol = request.form.get('Cholesterol')  

        # Convert categorical data into model-readable format  
        fever = "Yes" if fever.lower() == "yes" else "No"  
        cough = "Yes" if cough.lower() == "yes" else "No"  
        fatigue = "Yes" if fatigue.lower() == "yes" else "No"  
        difficulty = "Yes" if difficulty.lower() == "yes" else "No"  

        # Create a DataFrame with the correct column names  
        feature_dict = {  
            'Fever': [fever],  
            'Cough': [cough],  
            'Fatigue': [fatigue],  
            'Difficulty Breathing': [difficulty],  
            'Age': [age],  
            'Gender': [gender],  
            'Blood Pressure': [blood_pressure],  
            'Cholesterol Level': [cholesterol]  
        }  
        features = pd.DataFrame.from_dict(feature_dict)  

        # Predictions and probabilities for Model 1  
        prediction1_encoded = model1.predict(features)[0]  
        prediction1 = label_encoder1.inverse_transform([prediction1_encoded])[0]  
        probability1 = model1.predict_proba(features)[0].max()  

        # Predictions and probabilities for Model 2  
        prediction2_encoded = model2.predict(features)[0]  
        prediction2 = label_encoder2.inverse_transform([prediction2_encoded])[0]  
        probability2 = model2.predict_proba(features)[0].max()  

        # Prepare features for the third model  
        features_str = features.astype(str).apply(' '.join, axis=1)  
        features_vectorized = vectorizer3.transform(features_str)  
        prediction3 = model3.predict(features_vectorized)[0]  
        probability3 = model3.predict_proba(features_vectorized)[0].max()  

        # Combine predictions and probabilities  
        final_prediction = {  
            "Model 1": {"Disease": prediction1, "Probability": round(probability1, 2)},  
            "Model 2": {"Disease": prediction2, "Probability": round(probability2, 2)},  
            "Model 3": {"Disease": prediction3, "Probability": round(probability3, 2)},  
        }  

        return jsonify(final_prediction)  
    except Exception as e:  
        return jsonify({'error': str(e)})  

@app.route('/get_medications', methods=['POST'])  
def get_medications():  
    disease = request.form['disease'].strip().lower()   
    medications = medications_dict.get(disease, "No specific treatment found.")  
    
    if isinstance(medications, str):  
        medications_list = medications.split(', ')  
        formatted_medications = 'List of Medication(s) or Suggestions:\n' + '\n'.join(medications_list)  
    else:  
        formatted_medications = "No specific treatment found."  
    
    return jsonify({'medications': formatted_medications})  

if __name__ == '__main__':  
    app.run(debug=True)