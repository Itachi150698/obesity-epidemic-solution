# Necessary Libraries Importing
import joblib
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
# import warnings
# warnings.filterwarnings("ignore", category=UserWarning)


##############################################################################################################################################################################################

app = Flask(__name__)

#################################################################################################################################################################################################

# Load the model using joblib and pickle
model = joblib.load(open('cancer.pkl', 'rb'))
model1 = joblib.load(open('heart.pkl', 'rb'))
model2 = joblib.load(open('liver.pkl', 'rb'))
model3 = joblib.load(open('diabetes.pkl', 'rb'))
scaler = joblib.load('diabetes_scaler.pkl')
model4 = joblib.load(open('kidney.pkl', 'rb'))

###############################################################################################################################################################################################

# Html File routes
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/team')
def team():
    return render_template("team.html")

@app.route('/testimonial')
def testimonial():
    return render_template("testimonial.html")

@app.route('/why')
def why():
    return render_template("why.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/help')
def help():
    return render_template("help.html")

@app.route('/terms')
def terms():
    return render_template("tc.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/disindex")
def disindex():
    return render_template("disindex.html")

@app.route("/cancer")
def cancer():
    return render_template("cancer.html")

@app.route("/heart")
def heart():
    return render_template("heart.html")

@app.route("/kidney")
def kidney():
    return render_template("kidney.html")

@app.route("/liver")
def liver():
    return render_template("liver.html")

@app.route("/diabetes")
def diabetes():
    return render_template("diabetes.html")

##############################################################################################################################################################################################

# Kidney Disease prediction route
@app.route('/predictkidney', methods=['POST']) 
def predictkidney():
    if request.method == 'POST':
        # Extract user input from the form
        age = int(request.form['age'])
        blood_pressure = int(request.form['blood_pressure'])
        specific_gravity= float(request.form['specific_gravity'])
        albumin= float(request.form['albumin'])
        sugar= int(request.form['sugar'])
        blood_glucose_random= int(request.form['blood_glucose_random'])
        blood_urea= int(request.form['blood_urea'])
        serum_creatinine= float(request.form['serum_creatinine'])
        sodium= float(request.form['sodium'])
        potassium= float(request.form['potassium'])
        haemoglobin= float(request.form['haemoglobin'])
        packed_cell_volume= float(request.form['packed_cell_volume'])
        white_blood_cell_count= int(request.form['white_blood_cell_count'])
        red_blood_cell_count= float(request.form['red_blood_cell_count'])
        hypertension= (request.form['hypertension'])
        diabetes_mellitus= (request.form['diabetes_mellitus'])
            
            
        user_data = pd.DataFrame({
    'Age': [age],
    'Blood Pressure': [blood_pressure],
    'Specific Gravity': [specific_gravity],
    'Albumin': [albumin],
    'Sugar': [sugar],
    'Blood Glucose Random': [blood_glucose_random],
    'Blood Urea': [blood_urea],
    'Serum Creatinine': [serum_creatinine],
    'Sodium': [sodium],
    'Potassium': [potassium],
    'Haemoglobin': [haemoglobin],
    'Packed Cell Volume': [packed_cell_volume],
    'White Blood Cell Count': [white_blood_cell_count],
    'Red Blood Cell Count': [red_blood_cell_count],
    'Hypertension': [hypertension],
    'Diabetes Mellitus': [diabetes_mellitus]
})

        
        input_data_as_numpy_array = np.asarray(user_data).reshape(1, -1)

        

        # Perform kidney prediction using your trained model
        output = model4.predict(input_data_as_numpy_array)

        # Generate a Pandas report
        prediction_report = generate_pandas_report(user_data, output)

        # Pass the prediction, report, and user data to the template
        return render_template('kidney_result.html', prediction=output, prediction_report=prediction_report, user_data=user_data)



#############################################################################################################################################################################################

# Liver Disease prediction route
@app.route('/predictliver', methods=['POST']) 
def predictliver():
    if request.method == 'POST':
        # Extract user input from the form
        age = int(request.form['age'])
        gender = (request.form['gender'])
        total_bilirubin = float(request.form['total_bilirubin'])
        direct_bilirubin = float(request.form['direct_bilirubin'])
        alkaline_phosphotase = int(request.form['alkaline_phosphotase'])
        alamine_aminotransferase = int(request.form['alamine_aminotransferase'])
        aspartate_aminotransferase = int(request.form['aspartate_aminotransferase'])
        total_protiens = float(request.form['total_protiens'])
        albumin = float(request.form['albumin'])
        albumin_and_globulin_ratio = float(request.form['albumin_and_globulin_ratio']) 

        # Create a DataFrame with the user input
        user_data = pd.DataFrame({
            'Age': [age],
            'Gender': [gender],
            'Total Bilirubin': [total_bilirubin],
            'Direct Bilirubin': [direct_bilirubin],
            'Alkaline Phosphotase': [alkaline_phosphotase],
            'Alamine Aminotransferase': [alamine_aminotransferase],
            'Aspartate Aminotransferase': [aspartate_aminotransferase],
            'Total Proteins': [total_protiens],
            'Albumin': [albumin],
            'Albumin And Globulin Ratio': [albumin_and_globulin_ratio],
        })
        
        input_data_as_numpy_array = np.asarray(user_data).reshape(1, -1)
        
        # Perform liver prediction using your trained model
        output = model2.predict(input_data_as_numpy_array)

        # Generate a Pandas report
        prediction_report = generate_pandas_report(user_data, output)

        # Pass the prediction, report, and user data to the template
        return render_template('liver_result.html', prediction=output, prediction_report=prediction_report, user_data=user_data)

##############################################################################################################################################################################################

# Cancer prediction route
@app.route('/predict', methods=['POST']) 
def predict():
    if request.method == 'POST':
        # Extract user input from the form
        clump_thickness = int(request.form['clump_thickness'])
        uniform_cell_size = int(request.form['uniform_cell_size'])
        uniform_cell_shape = int(request.form['uniform_cell_shape'])
        marginal_adhesion = int(request.form['marginal_adhesion'])
        single_epithelial_size = int(request.form['single_epithelial_size'])
        bare_nuclei = int(request.form['bare_nuclei'])
        bland_chromatin = int(request.form['bland_chromatin'])
        normal_nucleoli = int(request.form['normal_nucleoli'])
        mitoses = int(request.form['mitoses'])

        # Create a DataFrame with the user input
        user_data = pd.DataFrame({
            'Clump Thickness': [clump_thickness],
            'Uniform Cell size': [uniform_cell_size],
            'Uniform Cell shape': [uniform_cell_shape],
            'Marginal Adhesion': [marginal_adhesion],
            'Single Epithelial Cell Size': [single_epithelial_size],
            'Bare Nuclei': [bare_nuclei],
            'Bland Chromatin': [bland_chromatin],
            'Normal Nucleoli': [normal_nucleoli],
            'Mitoses': [mitoses],
        })
        
        input_data_as_numpy_array = np.asarray(user_data).reshape(1, -1)
        
        # Perform cancer prediction using your trained model
        output = model.predict(input_data_as_numpy_array)

        # Generate a Pandas report
        prediction_report = generate_pandas_report(user_data, output)

        # Pass the prediction, report, and user data to the template
        return render_template('cancer_result.html', prediction=output, prediction_report=prediction_report, user_data=user_data)

###############################################################################################################################################################################################

# Heart Disease prediction route
@app.route('/predictheart', methods=['POST']) 
def predictheart():
    if request.method == 'POST':
        # Extract user input from the form
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        CP = int(request.form['CP'])
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        FBS = int(request.form['FBS'])
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        EXANG = int(request.form['EXANG'])
        oldpeak = float(request.form['oldpeak'])
        SLOPE = int(request.form['SLOPE'])
        CA = int(request.form['CA'])
        THAL = int(request.form['THAL'])

        # Create a DataFrame with the user input
        user_data = pd.DataFrame({
            'Age in Years': [age],
            'Sex': [sex],
            'CP': [CP],
            'Trest Bps': [trestbps],
            'Cholesterol': [chol],
            'FBS': [FBS],
            'RESTECG': [restecg],
            'Thalach': [thalach],
            'EXANG': [EXANG],
            'Old Peak': [oldpeak],
            'SLOPE': [SLOPE],
            'CA': [CA],
            'THAL': [THAL],
        })
        
        # Perform heart disease prediction using your trained model
        output = model1.predict(user_data)

        # Generate a Pandas report
        prediction_report = generate_pandas_report(user_data, output)

        # Pass the prediction, report, and user data to the template
        return render_template('heart_result.html', prediction=output, prediction_report=prediction_report, user_data=user_data)

##############################################################################################################################################################################################

# Diabetes prediction route
@app.route('/predictt', methods=['POST'])
def predictt():
    if request.method == 'POST':
        # Extract user input from the form
        pregnancies = int(request.form['pregnancies'])
        glucose = int(request.form['glucose'])
        bloodpressure = int(request.form['bloodpressure'])
        skinthickness = int(request.form['skinthickness'])
        insulin = int(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = int(request.form['age'])

        # Create a DataFrame with the user input
        user_data = pd.DataFrame({
            'Pregnancies': [pregnancies],
            'Glucose': [glucose],
            'BloodPressure': [bloodpressure],
            'SkinThickness': [skinthickness],
            'Insulin': [insulin],
            'Body Mass Index': [bmi],
            'Diabetes Pedigree Function': [dpf],
            'Age': [age]
        })
        
        # changing the input_data to numpy array
        input_data_as_numpy_array = np.asarray(user_data)
        
        # Scale the input data
        input_data_as_numpy_array = np.asarray(user_data).reshape(1, -1)
        # std_data = scaler.transform(input_data_as_numpy_array)

        # Perform diabetes prediction using your trained model
        output = model3.predict(input_data_as_numpy_array)

        # Generate a Pandas report
        prediction_report = generate_pandas_report(user_data, output)

        # Pass the prediction, report, and user data to the template
        return render_template('diab_result.html', prediction=output, prediction_report=prediction_report, user_data=user_data)
    
    print(model3.columns)

###############################################################################################################################################################################################

def generate_pandas_report(user_data, prediction):
    # Your actual report generation logic
    # This is a placeholder, replace it with the actual logic based on your requirements
    report_html = f"<p>User Data: {user_data.to_html()}</p><p>Prediction: {prediction}</p>"
    return report_html 

################################################################################################################################################################################################

if __name__ == '__main__':
    app.run(debug=True)


