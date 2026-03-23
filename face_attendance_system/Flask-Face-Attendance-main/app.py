import os
import cv2
import joblib
import shutil
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, send_file,abort,Response, session, redirect, url_for
from datetime import date, datetime
from sklearn.neighbors import KNeighborsClassifier
import threading
import time

# Set working directory to script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Initialize Flask App
app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = '12938'  # Change this to a secure key

# Global Constants
FACE_CASCADE_PATH = 'static/haarcascade_frontalface_default.xml'
MODEL_PATH = 'static/face_recognition_model.pkl'
FACES_DIR = 'static/faces'
ATTENDANCE_DIR = 'Attendance'

# Global variables for video streaming
camera = None
recognition_status = {'status': 'Scanning...', 'message': ''}
face_detected_time = None
recognition_in_progress = False

# Lock for thread-safe CSV operations
csv_lock = threading.Lock()

# Ensure necessary directories exist
os.makedirs(FACES_DIR, exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)

# Load Haarcascade model
if not os.path.exists(FACE_CASCADE_PATH):
    raise FileNotFoundError(f"Haarcascade file not found: {FACE_CASCADE_PATH}")
face_detector = cv2.CascadeClassifier(FACE_CASCADE_PATH)

# Ensure today's attendance file exists
attendance_file = f'{ATTENDANCE_DIR}/Attendance-{date.today().strftime("%m_%d_%y")}.csv'
if not os.path.exists(attendance_file):
    with open(attendance_file, 'w') as f:
        f.write('Name,Roll,Time\n')

# Helper Functions
def get_today_date():
    return date.today().strftime("%d-%B-%Y")

def get_total_registered_users():
    return len(os.listdir(FACES_DIR))

def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    return faces

def train_model():
    faces, labels = [], []
    for user in os.listdir(FACES_DIR):
        for img_name in os.listdir(f'{FACES_DIR}/{user}'):
            img = cv2.imread(f'{FACES_DIR}/{user}/{img_name}')
            if img is None:
                continue
            resized_face = cv2.resize(img, (50, 50)).ravel()
            faces.append(resized_face)
            labels.append(user)
    if faces:
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(np.array(faces), labels)
        joblib.dump(model, MODEL_PATH)
    else:
        # If no faces, remove the model file if it exists
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)

def add_attendance(name):
    """Add attendance for a user. Returns (True, message) if new, (False, message) if duplicate."""
    try:
        username, userid = name.split('_')
        userid = int(userid)
    except ValueError:
        return False, f"Invalid user format: {name}"
    current_time = datetime.now().strftime("%H:%M:%S")
    with csv_lock:
        try:
            df = pd.read_csv(attendance_file)
            if 'Roll' not in df.columns:
                df = pd.DataFrame(columns=['Name', 'Roll', 'Time'])
        except (FileNotFoundError, pd.errors.EmptyDataError):
            df = pd.DataFrame(columns=['Name', 'Roll', 'Time'])
        
        if userid not in df['Roll'].values:
            with open(attendance_file, 'a') as f:
                f.write(f'{username},{userid},{current_time}\n')
            return True, f"Attendance recorded for {name}"
        else:
            return False, f"Already marked - {name}"

def extract_attendance():
    with csv_lock:
        try:
            df = pd.read_csv(attendance_file)
            required_columns = ['Name', 'Roll', 'Time']
            df.columns = required_columns
            return df['Name'], df['Roll'], df['Time'], len(df)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            return [], [], [], 0

def get_camera_index():
    for i in range(2):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cap.release()
            return i
    return -1

# Flask Routes
@app.route('/')
def home():
    return render_template('home.html')

ATTENDANCE_DIR = os.path.join(os.getcwd(), "Attendance")

def fetch_all_attendance():
    """Fetch all attendance records from multiple CSV files and combine them into one DataFrame."""
    with csv_lock:
        if not os.path.exists(ATTENDANCE_DIR):
            return pd.DataFrame(columns=["#", "Name", "ID", "Time", "Date"])  # Empty table if no records exist

        all_data = []
        
        # Iterate over all CSV files in the Attendance directory
        for file in os.listdir(ATTENDANCE_DIR):
            if file.endswith(".csv"):
                file_path = os.path.join(ATTENDANCE_DIR, file)
                try:
                    df = pd.read_csv(file_path)
                    df["Date"] = file.replace("Attendance-", "").replace(".csv", "").replace("_", "/")  # Extract date
                    all_data.append(df)
                except pd.errors.EmptyDataError:
                    continue

        if all_data:
            return pd.concat(all_data, ignore_index=True)
        else:
            return pd.DataFrame(columns=["#", "Name", "ID", "Time", "Date"])  # Return empty DataFrame if no data

@app.route('/download')
def download():
    """Download all attendance records as a CSV file."""
    combined_file = os.path.join(ATTENDANCE_DIR, "All_Attendance_Records.csv")

    # Fetch all attendance data and save as CSV
    df = fetch_all_attendance()
    if df.empty:
        return abort(404, description="No attendance records available.")

    df.to_csv(combined_file, index=False)

    return send_file(combined_file, as_attachment=True, mimetype='text/csv')

def identify_face(face):
    if not os.path.exists(MODEL_PATH):
        return None  # No trained model available
    
    model = joblib.load(MODEL_PATH)  # Load trained model
    prediction = model.predict(face)  # Predict the user
    return prediction

# Train model on startup if faces exist
train_model()

def generate_frames():
    global recognition_in_progress
    
    cam_index = get_camera_index()
    if cam_index == -1:
        # Yield a static error frame
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, "No camera available", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        ret, buffer = cv2.imencode('.jpg', error_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        return
    
    cap = cv2.VideoCapture(cam_index)
    start_time = None
    found_face = False
    marked_users = {}  # Store user marking status
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        faces = extract_faces(frame)
        status_text = "Scanning for face..."
        status_color = (0, 255, 0)
        
        if faces is not None and len(faces) > 0:
            if start_time is None:
                start_time = time.time()
            
            found_face = True
            for (x, y, w, h) in faces:
                face_img = cv2.resize(frame[y:y+h, x:x+w], (50, 50)).reshape(1, -1)
                identity = identify_face(face_img)
                
                if identity is not None:
                    user = identity[0]
                    
                    # Check if user already marked in this session
                    if user in marked_users:
                        status_text = f"Already marked: {user}"
                        status_color = (0, 165, 255)  # Orange
                    else:
                        # Try to add attendance
                        success, message = add_attendance(user)
                        marked_users[user] = success
                        
                        if success:
                            status_text = f"✓ {message}"
                            status_color = (0, 255, 0)  # Green
                        else:
                            status_text = f"⚠ {message}"
                            status_color = (0, 165, 255)  # Orange
                    
                    cv2.putText(frame, status_text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), status_color, 2)
                    recognition_in_progress = False
                else:
                    status_text = "Face not recognized"
                    status_color = (0, 0, 255)  # Red
                    cv2.putText(frame, status_text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), status_color, 2)
                    
                    # Check 5-second timeout
                    if start_time and (time.time() - start_time) > 5:
                        status_text = "Not recognized - Please register"
                        recognition_in_progress = False
        else:
            start_time = None
            found_face = False
        
        # Add status text at bottom
        cv2.putText(frame, status_text, (30, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.05)  # Small delay to reduce CPU usage

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/add', methods=['POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    user_folder = f'{FACES_DIR}/{newusername}_{newuserid}'
    os.makedirs(user_folder, exist_ok=True)
    cam_index = get_camera_index()
    if cam_index == -1:
        return render_template('home.html', mess='No camera available for face capture.')
    cap = cv2.VideoCapture(cam_index)
    count = 0
    start_time = time.time()
    while count < 50 and (time.time() - start_time) < 10:  # Capture for up to 10 seconds
        ret, frame = cap.read()
        if not ret:
            break
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            if count >= 50:
                break
            face_img = frame[y:y+h, x:x+w]
            img_path = f'{user_folder}/{newusername}_{count}.jpg'
            cv2.imwrite(img_path, face_img)
            count += 1
    cap.release()
    if count == 0:
        return render_template('home.html', mess='No faces detected during capture.')
    train_model()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l,
                           totalreg=get_total_registered_users(), datetoday2=get_today_date())

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    if username == 'Admin' and password == 'Admin@123':
        session['admin'] = True
        return redirect(url_for('admin'))
    else:
        return render_template('home.html', mess='Invalid credentials')

@app.route('/admin')
def admin():
    if not session.get('admin'):
        return redirect(url_for('home'))
    names, rolls, times, l = extract_attendance()
    return render_template('admin.html', names=names, rolls=rolls, times=times, l=l, 
                           totalreg=get_total_registered_users(), datetoday2=get_today_date(), users=get_users())

@app.route('/logout')
def logout():
    session.pop('admin', None)
    return redirect(url_for('home'))

def get_users():
    users = []
    for user in os.listdir(FACES_DIR):
        if os.path.isdir(os.path.join(FACES_DIR, user)):
            username, userid = user.rsplit('_', 1)
            users.append({'name': username, 'id': userid})
    return users

@app.route('/remove', methods=['POST'])
def remove():
    if not session.get('admin'):
        return redirect(url_for('home'))
    userid = request.form['userid']
    user_folder = None
    for user in os.listdir(FACES_DIR):
        if user.endswith(f'_{userid}'):
            user_folder = os.path.join(FACES_DIR, user)
            break
    if user_folder and os.path.exists(user_folder):
        shutil.rmtree(user_folder)
        train_model()  # Retrain model after removing user
    return redirect(url_for('admin'))

@app.route('/delete_attendance', methods=['POST'])
def delete_attendance():
    if not session.get('admin'):
        return redirect(url_for('home'))
    index = int(request.form['index'])
    with csv_lock:
        try:
            df = pd.read_csv(attendance_file)
            if 0 <= index < len(df):
                df = df.drop(index)
                df.to_csv(attendance_file, index=False)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            pass
    return redirect(url_for('admin'))

if __name__ == '__main__':
    app.run(debug=True)