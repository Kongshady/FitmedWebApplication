from flask import Flask, render_template, request, jsonify  
import joblib  
import pandas as pd  
 
app = Flask(__name__)  

model_path = "models/disease_model.pkl"
vectorizer_path = "models/vectorizer.pkl"  
model_pipeline = joblib.load(model_path)  
label_encoder = joblib.load(vectorizer_path)  


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
        form_data = {  
            'Fever': request.form.get('Fever'),  
            'Cough': request.form.get('Cough'),  
            'Fatigue': request.form.get('Fatigue'),  
            'Difficulty Breathing': request.form.get('Difficulty'),  
            'Age': float(request.form.get('Age')),  
            'Gender': "Male" if request.form.get('Gender').lower() == "male" else "Female",  
            'Blood Pressure': request.form.get('BloodPressure'),  
            'Cholesterol Level': request.form.get('Cholesterol')  
        }  
 
        for symptom in ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing']:  
            form_data[symptom] = "Yes" if form_data[symptom].lower() == "yes" else "No"  

        features = pd.DataFrame([form_data])  

        prediction_encoded = model_pipeline.predict(features)[0]  
        
        prediction = label_encoder.inverse_transform([prediction_encoded])[0]  

        return jsonify({'prediction': prediction})  
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