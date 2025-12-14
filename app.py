# app.py
from flask import Flask, render_template, request, redirect, url_for, send_file
import pandas as pd
import os
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np # Added for helper functions

app = Flask(__name__)
DATA_PATH = "students.csv"
INTERVENTION_LOG = "interventions.csv"

# In-memory 'cache' for the trained model
global_model_data = {}

# Utility: load dataset (if absent, create empty df with columns)
def load_students():
    if os.path.exists(DATA_PATH):
        try:
            df = pd.read_csv(DATA_PATH)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame(columns=[
                "student_id","name","attendance_percent","avg_assignment_score",
                "study_hours_per_week","behavior_incidents","previous_grade","label"
            ])
    else:
        df = pd.DataFrame(columns=[
            "student_id","name","attendance_percent","avg_assignment_score",
            "study_hours_per_week","behavior_incidents","previous_grade","label"
        ])
    return df

# Train decision tree on CSV data
def train_tree():
    df = load_students()
    if df.shape[0] < 5:
        return None, "Not enough data to train (need >=5 rows)."
    
    # Define feature columns
    feature_cols = ["attendance_percent","avg_assignment_score","study_hours_per_week","behavior_incidents","previous_grade"]
    
    # Ensure all feature columns are numeric
    try:
        X = df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    except Exception as e:
        return None, f"Error converting features to numbers: {e}. Check your CSV."
        
    y = df["label"].astype(str)
    
    # Check if there's data to train on
    if y.empty:
        return None, "No labels found in data."

    # no categorical encoders needed here (all numeric); if label is categorical, encode
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    if len(le.classes_) <= 1:
        return None, f"Model training failed (only 1 class found in data: {le.classes_}). Add more varied data."

    # --- MODIFICATION 1: Add stratify=y_enc ---
    # This ensures the train/test split has a proportional number of "OnTrack" and "AtRisk" students.
    # This is the most likely fix for your model always predicting "OnTrack".
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
    except ValueError as e:
        # This can happen if one class has only 1 member
        return None, f"Stratify failed. Your dataset is too small or imbalanced to split. Error: {e}"

    # --- MODIFICATION 2: Add check for classes in the training data ---
    if len(np.unique(y_train)) <= 1:
        return None, "Model training failed (only 1 class in training split). Your dataset is likely too small or imbalanced even after stratification."

    clf = DecisionTreeClassifier(random_state=42, max_depth=4)
    clf.fit(X_train, y_train)
    
    test_score = clf.score(X_test, y_test)
    print(f"Model trained. Test accuracy: {test_score}")
    
    model_data = {
        "model": clf,
        "le": le,
        "features": feature_cols,
        "test_score": test_score,
        "tree_text": export_text(clf, feature_names=feature_cols, class_names=list(le.classes_))
    }
    
    # Cache the model
    global_model_data.clear()
    global_model_data.update(model_data)
    return model_data, None # No error

# --- ADDED FUNCTION: This was missing but implied by your code ---
def suggest_interventions(input_data, model_data, predicted_label, probs):
    """
    Generates a dictionary with prediction results and suggested interventions.
    """
    suggestions = []
    le = model_data["le"]
    
    at_risk_prob = 0.0
    try:
        # Find which index in le.classes_ corresponds to "AtRisk"
        classes_list = list(le.classes_)
        if "AtRisk" in classes_list:
            at_risk_idx = classes_list.index("AtRisk")
            if probs is not None and len(probs) > at_risk_idx:
                at_risk_prob = probs[at_risk_idx]
    except Exception as e:
        print(f"Error finding AtRisk prob: {e}")

    # Confidence is the probability of the *predicted* class
    confidence = 0.5 # Default confidence if no proba
    try:
        pred_idx = list(le.classes_).index(predicted_label)
        if probs is not None:
            confidence = probs[pred_idx]
    except Exception:
        pass # Keep default confidence

    # --- Intervention Logic ---
    # Always check for triggers, even if "OnTrack", if "AtRisk" prob is notable
    check_triggers = False
    if predicted_label == "AtRisk":
        check_triggers = True
        suggestions.append("Predicted AtRisk: Student needs immediate review.")
    elif at_risk_prob > 0.3: # e.g., OnTrack but 35% chance of AtRisk
        check_triggers = True
        suggestions.append("Flagged: Predicted OnTrack, but 'AtRisk' probability is high.")

    if check_triggers:
        try:
            if float(input_data["attendance_percent"]) < 80:
                suggestions.append("Low attendance: Schedule parent meeting & attendance monitoring.")
            if float(input_data["avg_assignment_score"]) < 70:
                suggestions.append("Low assignment scores: Arrange targeted tutoring and assignment walkthroughs.")
            if float(input_data["study_hours_per_week"]) < 5:
                suggestions.append("Low study hours: Provide time-management workshop and study plan.")
            if float(input_data["behavior_incidents"]) > 0:
                suggestions.append("Behavior incidents: Behavioural counselor meeting and one-on-one mentoring.")
            if float(input_data["previous_grade"]) < 65:
                suggestions.append("Previously low grades: Remedial classes and frequent short formative quizzes.")
        except Exception as e:
            suggestions.append(f"Error checking triggers: {e}")


    if not suggestions:
        suggestions.append("No specific flags — continue monitoring. Recommend enrichment activities.")

    return {
        "predicted": predicted_label,
        "confidence": f"{confidence:.2f}",
        "suggestions": suggestions
    }

# --- ADDED FUNCTION: This was missing but implied by interventions.csv ---
def log_intervention(student_id, name, result_dict):
    """
    Appends a record to the interventions.csv log file.
    """
    try:
        log_df = pd.DataFrame({
            "student_id": [student_id],
            "name": [name],
            "predicted": [result_dict["predicted"]],
            "confidence": [result_dict["confidence"]],
            "suggestions": ["|".join(result_dict["suggestions"])] # Join list with a separator
        })
        
        # Check if file exists to write header or not
        file_exists = os.path.exists(INTERVENTION_LOG)
        
        log_df.to_csv(INTERVENTION_LOG, mode='a', header=not file_exists, index=False)
        
    except Exception as e:
        print(f"Error logging intervention: {e}")

# --- Routes ---

@app.route("/")
def index():
    df = load_students()
    return render_template("index.html", 
                           n_students=df.shape[0], 
                           model_trained=bool(global_model_data),
                           err=request.args.get("err"))

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == '':
            return redirect(url_for("index", err="No file selected"))
        if file and file.filename.endswith(".csv"):
            file.save(DATA_PATH)
            # Clear old model, as data has changed
            global_model_data.clear()
            return redirect(url_for("index"))
    return render_template("upload.html")

@app.route("/students")
def students():
    df = load_students()
    return render_template("students.html", tables=[df.to_html(classes='table', index=False)])

@app.route("/train")
def train():
    # Clear old model first
    global_model_data.clear()
    model, err = train_tree()
    if err:
        return redirect(url_for("index", err=err))
    
    # Display the tree
    tree_text = global_model_data.get("tree_text", "Could not export tree text.")
    return render_template("train.html", tree_text=tree_text)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if not global_model_data:
        return render_template("predict.html", error="Model not trained yet. Please train model first.", trained=False)

    model_data = global_model_data
    
    if request.method == "POST":
        try:
            input_data = {
                "attendance_percent": request.form.get("attendance_percent", 0),
                "avg_assignment_score": request.form.get("avg_assignment_score", 0),
                "study_hours_per_week": request.form.get("study_hours_per_week", 0),
                "behavior_incidents": request.form.get("behavior_incidents", 0),
                "previous_grade": request.form.get("previous_grade", 0)
            }
            student_id = request.form.get("student_id", "")
            name = request.form.get("name", "")

            # Ensure features are in the correct order
            X = [[ float(input_data[feat]) for feat in model_data["features"] ]]
            
            clf = model_data["model"]
            le = model_data["le"]
            
            probs = clf.predict_proba(X)[0]
            pred_enc = clf.predict(X)[0]
            pred_label = le.inverse_transform([pred_enc])[0]

            suggestions_dict = suggest_interventions(input_data, model_data, pred_label, probs)
            
            # --- ADDED: Call to log the intervention ---
            log_intervention(student_id, name, suggestions_dict)

            return render_template("predict.html", trained=True, result=suggestions_dict)
        except Exception as e:
            return render_template("predict.html", trained=True, error=f"Prediction failed: {e}")

    return render_template("predict.html", trained=True, error=None)

@app.route("/whatif", methods=["GET", "POST"])
def whatif():
    if not global_model_data:
        return render_template("whatif.html", error="Model not trained yet. Please train model first.", trained=False)
    
    model_data = global_model_data
    
    if request.method == "POST":
        try:
            # We get the form data as a dictionary
            input_data = request.form.to_dict()

            X = [[ float(input_data[feat]) for feat in model_data["features"] ]]
            
            clf = model_data["model"]
            le = model_data["le"]
            
            probs = clf.predict_proba(X)[0] if hasattr(clf, "predict_proba") else None
            pred_enc = clf.predict(X)[0]
            pred_label = le.inverse_transform([pred_enc])[0]
            
            suggestions = suggest_interventions(input_data, model_data, pred_label, probs)
            
            # --- MODIFICATION 3: Pass input_data as 'defaults' to keep form values sticky ---
            # The template will use 'defaults' to pre-fill the form
            return render_template("whatif.html", trained=True, result=suggestions, defaults=input_data)
        except Exception as e:
            return render_template("whatif.html", trained=True, error=f"Simulation failed: {e}")

    # GET -> sample defaults
    defaults = {"attendance_percent": 80, "avg_assignment_score": 70, "study_hours_per_week": 6, "behavior_incidents": 0, "previous_grade": 75}
    return render_template("whatif.html", trained=True, defaults=defaults)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)
