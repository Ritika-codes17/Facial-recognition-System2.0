import streamlit as st
import cv2
import numpy as np
from PIL import Image
from collections import deque, Counter # Imported for the Voting System

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Facial Emotion Recognition", page_icon="🙂", layout="wide")

# --- CUSTOM CSS (Gray & Violet Theme) ---
st.markdown("""
<style>
    /* Hide Top Bar */
    header {visibility: hidden;}
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    
    /* Main Background */
    .stApp { background: linear-gradient(to bottom right, #E0E0E0, #FAFAFA); }
    
    /* Violet Button Styling */
    div.stButton > button {
        width: 100%; height: 50px; font-size: 20px; font-weight: bold;
        background-color: #6A1B9A; color: white; border-radius: 8px; border: none;
        box-shadow: 0px 4px 6px rgba(0,0,0,0.1); transition: 0.3s;
    }
    div.stButton > button:hover { background-color: #4A148C; transform: translateY(-2px); }

    /* Emotion Box Styling */
    .emotion-box {
        padding: 15px; border-radius: 15px; text-align: center;
        background-color: #FFFFFF; box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-top: 10px; border: 2px solid #6A1B9A;
    }
    .emotion-emoji { font-size: 80px; }
    .emotion-text { font-size: 28px; font-weight: bold; color: #333; font-family: sans-serif; }
    
    .welcome-text { text-align: center; color: #616161; font-size: 18px; font-weight: 500; margin-bottom: -15px; }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if 'mode' not in st.session_state:
    st.session_state['mode'] = 'Live Feed'

# --- TITLE ---
st.markdown("<p class='welcome-text'>Welcome to Mini Project</p>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: #212121; margin-bottom: 20px;'>Facial Emotion Recognition</h1>", unsafe_allow_html=True)

# --- NAVIGATION ---
_, col_nav1, col_nav2, _ = st.columns([1, 2, 2, 1])
with col_nav1:
    if st.button("📹 Live Video Feed", use_container_width=True):
        st.session_state['mode'] = 'Live Feed'
with col_nav2:
    if st.button("📸 Capture Snapshot", use_container_width=True):
        st.session_state['mode'] = 'Snapshot'
st.write("---")

# --- LAZY LOAD RESOURCES ---
@st.cache_resource
def load_resources():
    with st.spinner("Loading AI Brain..."):
        import tensorflow as tf
        model = tf.keras.models.load_model('emotion_detection_model.h5', compile=False)
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        return model, facecasc

try:
    model, facecasc = load_resources()
    emotion_dict = {
        0: ("Angry", "😡"), 1: ("Disgusted", "🤢"), 2: ("Fearful", "😨"), 
        3: ("Happy", "😀"), 4: ("Neutral", "😐"), 5: ("Sad", "😢"), 6: ("Surprised", "😲")
    }
except Exception as e:
    st.error(f"Error loading resources: {e}")
    st.stop()

# --- HELPER: EMOTION BOX ---
def display_emotion_box(emotion_label, emoji):
    st.markdown(f"""
        <div class="emotion-box">
            <div class="emotion-emoji">{emoji}</div>
            <div class="emotion-text">{emotion_label}</div>
        </div>
    """, unsafe_allow_html=True)

# --- APP MODES ---

# ----------------- LIVE FEED (WITH VOTING SYSTEM) -----------------
if st.session_state['mode'] == 'Live Feed':
    c1, c2, c3 = st.columns([1, 6, 1])
    with c2:
        st.subheader("🔴 Live Feed")
        run = st.checkbox('Start Camera', value=False)
        
        col_video, col_result = st.columns([2, 1])
        
        with col_video:
            FRAME_WINDOW = st.image([])
        with col_result:
            EMOTION_BOX = st.empty()

        if run:
            cap = cv2.VideoCapture(0)
            
            # --- VOTING SYSTEM VARIABLES ---
            # We will store the last 10 predictions here
            emotion_window = deque(maxlen=10) 
            
            while run:
                ret, frame = cap.read()
                if not ret: 
                    st.error("Camera disconnected.")
                    break
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
                
                prediction_made = False
                final_label = None
                final_emoji = None

                for (x, y, w, h) in faces:
                    cv2.rectangle(frame_rgb, (x, y-50), (x+w, y+h+10), (154, 27, 106), 3)
                    
                    # Process every frame for smoothness
                    roi_gray = gray[y:y + h, x:x + w]
                    roi_gray_resized = cv2.resize(roi_gray, (48, 48))
                    roi_gray_normalized = roi_gray_resized / 255.0
                    cropped_img = np.expand_dims(np.expand_dims(roi_gray_normalized, -1), 0)
                    
                    # Predict
                    prediction = model.predict(cropped_img, verbose=0)
                    maxindex = int(np.argmax(prediction))
                    
                    # Add prediction to our voting window
                    emotion_window.append(maxindex)
                    prediction_made = True
                    
                    # --- THE VOTE ---
                    # Find the most common emotion in the last 10 frames
                    if len(emotion_window) > 0:
                        vote_result = Counter(emotion_window).most_common(1)[0][0]
                        final_label, final_emoji = emotion_dict[vote_result]
                        
                        # Draw label on video
                        cv2.putText(frame_rgb, final_label, (x+10, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    break # Only process one face
                
                # If no face is found, clear the voting history so old emotions don't stick
                if not prediction_made:
                    emotion_window.clear()

                # Update Screen
                FRAME_WINDOW.image(frame_rgb, use_container_width=True)
                
                if final_label and final_emoji:
                    with EMOTION_BOX.container():
                        display_emotion_box(final_label, final_emoji)
                else:
                    EMOTION_BOX.empty()
                
            cap.release()
        else:
            FRAME_WINDOW.info("Click 'Start Camera' to begin.")

# ----------------- SNAPSHOT MODE -----------------
elif st.session_state['mode'] == 'Snapshot':
    c1, c2, c3 = st.columns([1, 6, 1])
    with c2:
        st.subheader("📸 Snapshot Analysis")
        img_file_buffer = st.camera_input("Smile and click!")
        
        if img_file_buffer is not None:
            bytes_data = img_file_buffer.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
            annotated_image, label, emoji = None, None, None

            # Logic to process snapshot
            gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
            faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                cv2.rectangle(cv2_img, (x, y-50), (x+w, y+h+10), (154, 27, 106), 3)
                roi_gray = gray[y:y + h, x:x + w]
                roi_gray_resized = cv2.resize(roi_gray, (48, 48))
                roi_gray_normalized = roi_gray_resized / 255.0
                cropped_img = np.expand_dims(np.expand_dims(roi_gray_normalized, -1), 0)
                
                prediction = model.predict(cropped_img, verbose=0)
                maxindex = int(np.argmax(prediction))
                label, emoji = emotion_dict[maxindex]
                cv2.putText(cv2_img, label, (x+10, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                break
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(cv2_img, use_container_width=True, caption="Analyzed Photo")
            with col2:
                if label and emoji:
                    display_emotion_box(label, emoji)
                else:
                    st.warning("No face found!")