from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html") 
   
@app.route('/contact')
def contact():
    return render_template("contact.html")

@app.route('/login')
def login():
    return render_template("login.html")

@app.route('/signup')
def signup():
    return render_template("signup.html")

@app.route('/signup-success')
def signup_success():
    return render_template("signup_success.html")


@app.route('/testing')
def testing():
    return render_template("testing.html")

@app.route('/predict', methods=['POST'])
def predict():

    features = [[
        float(request.form['age']),
        float(request.form['sex']),
        float(request.form['cp']),
        float(request.form['trestbps']),
        float(request.form['chol']),
        float(request.form['fbs']),
        float(request.form['restecg']),
        float(request.form['thalach']),
        float(request.form['exang']),
        float(request.form['oldpeak']),
        float(request.form['slope']),
        float(request.form['ca']),
        float(request.form['thal'])
    ]]

    prediction = model.predict(features)[0]

    if prediction == 1:
        return render_template("result_yes.html")
    else:
        return render_template("result_no.html")

if __name__ == "__main__":
    app.run(debug=True)
    

