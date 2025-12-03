import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import joblib
from PIL import Image
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
from collections import deque

# --- Page Configuration ---
st.set_page_config(page_title="AI Real-time Posture Correction", page_icon="🐢")

# 스타일 설정
st.markdown("""
    <style>
    .big-font { font-size:24px !important; font-weight: bold; }
    .good-text { color: #2ecc71; font-weight: bold; font-size: 20px;}
    .mild-text { color: #f1c40f; font-weight: bold; font-size: 20px;}
    .severe-text { color: #e74c3c; font-weight: bold; font-size: 20px;}
    .warning-box { background-color: #fadbd8; border: 2px solid #e74c3c; padding: 15px; border-radius: 10px; text-align: center; }
    </style>
    """, unsafe_allow_html=True)

st.title("🐢 AI Real-time Turtle Neck Diagnosis")
st.write("Turn on the webcam to analyze your posture in real-time.")

# --- Load Model & MediaPipe ---
@st.cache_resource
def load_model():
    try:
        return joblib.load('posture_model.pkl')
    except:
        return None

model = load_model()
mp_pose = mp.solutions.pose

# --- Helper Function: Adjust Probabilities ---
def adjust_probabilities(probs, classes):
    """
    Severe 확률을 낮추고 나머지를 다시 정규화하는 보정 함수
    probs: numpy array (각 클래스 확률)
    classes: model.classes_ (['good','mild','severe'] 등)
    """
    prob_dict = {cls: float(p) for cls, p in zip(classes, probs)}

    # severe 확률 0.7배로 줄이기 (너무 예민하게 뜨는 것 방지)
    if 'severe' in prob_dict:
        prob_dict['severe'] *= 0.7

    # 합이 1이 되도록 다시 정규화
    total = sum(prob_dict.values())
    if total > 0:
        for cls in prob_dict:
            prob_dict[cls] /= total

    # 보정 후 가장 높은 클래스
    new_pred = max(prob_dict, key=prob_dict.get)
    return prob_dict, new_pred


# --- Real-time Video Processing Class ---
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
        self.model = model
        # 결과 공유를 위한 변수
        self.latest_probs = {'good': 0, 'mild': 0, 'severe': 0}
        self.latest_pred = None
        # 최근 프레임 확률을 저장해서 smoothing에 사용
        self.history = deque(maxlen=10)   # ← 추가


    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # 1. Image Processing
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            try:
                # 2. Feature Extraction
                l_sh = landmarks[11]; r_sh = landmarks[12]
                center_x = (l_sh.x + r_sh.x) / 2
                center_y = (l_sh.y + r_sh.y) / 2
                width = np.linalg.norm(np.array([l_sh.x, l_sh.y]) - np.array([r_sh.x, r_sh.y]))
                if width == 0: width = 1

                indices = [0, 2, 5, 7, 8, 11, 12]
                features = []
                
                h, w, _ = img.shape
                draw_points = []

                for idx in indices:
                    lm = landmarks[idx]
                    norm_x = (lm.x - center_x) / width
                    norm_y = (lm.y - center_y) / width
                    features.extend([norm_x, norm_y])
                    draw_points.append((int(lm.x * w), int(lm.y * h)))

                # 3. Prediction
                if self.model:
                    # 1) 현재 프레임 확률
                    probs = self.model.predict_proba([features])[0]   # shape: (n_classes,)
                    self.history.append(probs)

                    # 2) 최근 프레임 평균으로 smoothing
                    avg_probs = np.mean(self.history, axis=0)
                    classes = self.model.classes_

                    # 3) severe 확률 보정 + 정규화
                    final_prob_dict, final_pred = adjust_probabilities(avg_probs, classes)

                    # 4) 공유 변수 업데이트 (UI에서 사용)
                    self.latest_probs = final_prob_dict      # 예: {'good':0.6,'mild':0.3,'severe':0.1}
                    self.latest_pred = final_pred

                # 4. 화면에는 점만 찍기 (텍스트 없음)
                for px, py in draw_points:
                    cv2.circle(img, (px, py), 5, (0, 255, 0), -1)

                    
            except Exception as e:
                pass

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- Main Tab Configuration ---
tab1, tab2 = st.tabs(["📷 Real-time Analysis", "🖼️ Upload Photo"])

# Tab 1: Real-time with External UI
with tab1:
    st.header("Real-time Webcam")
    
    if model is None:
        st.error("Model file (posture_model.pkl) is missing.")
    else:
        col1, col2 = st.columns([2, 1])
        
        # -------------------- LEFT SIDE (VIDEO) --------------------
        with col1:
            ctx = webrtc_streamer(
                key="posture-check",
                video_processor_factory=VideoProcessor,
                mode=WebRtcMode.SENDRECV,
                media_stream_constraints={"video": True, "audio": False},
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                async_processing=True
            )

        # -------------------- RIGHT SIDE (UI STATUS) --------------------
        with col2:
            st.subheader("Live Status")
            status_text_ph = st.empty()

            st.write("**Prediction Confidence:**")

            # --- Row 1: Labels (Good / Mild / Severe) ---
            label_good, label_mild, label_severe = st.columns(3)

            with label_good:
                st.markdown(
                    "<p style='text-align: center; color: #2ecc71; font-weight: bold;'>Good</p>",
                    unsafe_allow_html=True
                )

            with label_mild:
                st.markdown(
                    "<p style='text-align: center; color: #f1c40f; font-weight: bold;'>Mild</p>",
                    unsafe_allow_html=True
                )

            with label_severe:
                st.markdown(
                    "<p style='text-align: center; color: #e74c3c; font-weight: bold;'>Severe</p>",
                    unsafe_allow_html=True
                )

            # --- Row 2: Full-width horizontal bars ---
            st.write("Good:")
            bar_good_ph = st.empty()

            st.write("Mild:")
            bar_mild_ph = st.empty()

            st.write("Severe:")
            bar_severe_ph = st.empty()

            # Warning box placeholder
            warning_ph = st.empty()


    # -------------------- LOOP (OUTSIDE col1/col2!) --------------------
    if ctx.state.playing:
        while True:
            if ctx.video_processor:
                probs = ctx.video_processor.latest_probs
                pred = ctx.video_processor.latest_pred

                if pred:
                    p_good = int(probs.get('good', 0) * 100)
                    p_mild = int(probs.get('mild', 0) * 100)
                    p_severe = int(probs.get('severe', 0) * 100)

                    # Status Text
                    if pred == 'good':
                        status_text_ph.markdown(
                            "<p class='good-text'>Status: GOOD 😊</p>",
                            unsafe_allow_html=True
                        )
                    elif pred == 'mild':
                        status_text_ph.markdown(
                            "<p class='mild-text'>Status: MILD 😐</p>",
                            unsafe_allow_html=True
                        )
                    else:
                        status_text_ph.markdown(
                            "<p class='severe-text'>Status: SEVERE 🐢</p>",
                            unsafe_allow_html=True
                        )

                    # Progress bars
                    bar_good_ph.progress(p_good, text=f"Good: {p_good}%")
                    bar_mild_ph.progress(p_mild, text=f"Mild: {p_mild}%")
                    bar_severe_ph.progress(p_severe, text=f"Severe: {p_severe}%")

                    # Warning box
                    if pred == 'severe':
                        warning_ph.markdown(
                            """
                            <div class='warning-box'>
                                🚨 <b>BAD POSTURE DETECTED!</b><br>
                                Please straighten your neck.
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        warning_ph.empty()

            import time
            time.sleep(0.1)

# Tab 2: Upload
with tab2:
    st.header("File Upload Diagnosis")
    uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file and model:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        img_np = np.array(image.convert('RGB'))
        pose_static = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=1)
        results = pose_static.process(img_np)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            try:
                l_sh = landmarks[11]; r_sh = landmarks[12]
                center_x = (l_sh.x + r_sh.x) / 2
                center_y = (l_sh.y + r_sh.y) / 2
                width = np.linalg.norm(np.array([l_sh.x, l_sh.y]) - np.array([r_sh.x, r_sh.y]))
                if width == 0: width = 1
                
                features = []
                indices = [0, 2, 5, 7, 8, 11, 12]
                for idx in indices:
                    lm = landmarks[idx]
                    features.extend([(lm.x - center_x)/width, (lm.y - center_y)/width])
                
                                # 3) 모델 예측 + severe 보정
                probs = model.predict_proba([features])[0]   # 0~1 확률
                classes = model.classes_

                # adjust_probabilities로 severe 확률 보정 + 정규화
                prob_dict_raw, pred = adjust_probabilities(probs, classes)
                # prob_dict_raw: {'good':0.6, 'mild':0.3, 'severe':0.1} 이런 형태

                # UI용 퍼센트 값으로 변환
                good_pct = prob_dict_raw.get('good', 0) * 100
                mild_pct = prob_dict_raw.get('mild', 0) * 100
                severe_pct = prob_dict_raw.get('severe', 0) * 100

                st.subheader("Analysis Result")

                st.write(f"**Good: {good_pct:.1f}%**")
                st.progress(int(good_pct))

                st.write(f"**Mild: {mild_pct:.1f}%**")
                st.progress(int(mild_pct))

                st.write(f"**Severe: {severe_pct:.1f}%**")
                st.progress(int(severe_pct))

                # pred는 보정된 확률 기준 (adjust_probabilities에서 온 값)
                if pred == 'severe':
                    st.error("🚨 WARNING: Severe Forward Head Posture detected!")
                elif pred == 'mild':
                    st.warning("🟡 Caution: Mild Forward Head Posture.")
                else:
                    st.success("🟢 Good Posture!")

            except:
                st.error("Analysis failed.")
        else:
            st.error("Person not found.")





