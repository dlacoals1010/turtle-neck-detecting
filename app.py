import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import av
import time
from collections import deque
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode, RTCConfiguration

# --- Page Configuration ---
st.set_page_config(page_title="AI Posture Correction Pro", page_icon="🐢", layout="wide")

# --- JS & CSS (audio + TTS + 스타일) ---
def get_audio_html():
    js_code = """
        <script>
        // Simple beep alert (for SEVERE)
        function playAlert() {
            var audio = new Audio('https://actions.google.com/sounds/v1/alarms/beep_short.ogg');
            audio.volume = 0.5;
            audio.play();
        }

        // ---- Voice guidance using Web Speech API ----
        window.lastPostureStatus = null;

        function speakPostureStatus(status) {
            if (!('speechSynthesis' in window)) return;

            var text = '';
            if (status === 'GOOD') {
                text = 'Posture is good.';
            } else if (status === 'MILD') {
                text = 'Posture is mild. Please lift your head slightly and relax your shoulders.';
            } else if (status === 'SEVERE') {
                text = 'Posture is severe. Pull your chin back and open your chest.';
            }
            if (text === '') return;

            var utterance = new SpeechSynthesisUtterance(text);
            utterance.lang = 'en-US';
            window.speechSynthesis.cancel();
            window.speechSynthesis.speak(utterance);
        }

        // Called from Python when posture class changes
        function updatePostureStatus(status) {
            if (window.lastPostureStatus === status) return;
            window.lastPostureStatus = status;
            speakPostureStatus(status);
        }
        </script>
        <div id="audio-container"></div>
    """
    return js_code

st.markdown("""
    <style>
    .good-text { color: #2ecc71; font-weight: bold; font-size: 30px; }
    .mild-text { color: #f1c40f; font-weight: bold; font-size: 30px; }
    .severe-text { color: #e74c3c; font-weight: bold; font-size: 30px; animation: blink 1s infinite; }
    
    .advice-box {
        background-color: #fff9c4;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #fbc02d;
        font-size: 18px;
        font-weight: bold;
        color: #333;
        margin-top: 10px;
    }

    @keyframes blink {
        50% { opacity: 0.5; }
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown(get_audio_html(), unsafe_allow_html=True)

st.title("🐢 AI Posture Correction Pro")
st.markdown(
    "Turn on your webcam to analyze your posture in real time. "
    "**Set your own best upright posture as the personal standard.**"
)

mp_pose = mp.solutions.pose

# ✅ NEW: static 이미지용 Pose 객체 (사진 분석용)
pose_static = mp_pose.Pose(
    static_image_mode=True,
    min_detection_confidence=0.5,
    model_complexity=1
)

# --- Distance → Probabilities (Good / Mild / Severe) ---
def distance_to_probs(distance, t_good=0.12, t_mild=0.23):
    """
    Map distance from baseline to probabilities for good/mild/severe.
    t_good: below this, mostly 'good'
    t_mild: above this, starts to become 'severe'
    """
    d = float(distance)

    # Good score decreases as distance grows
    good_score = max(0.0, 1.0 - d / max(t_good, 1e-6))

    # Mild score is high in the middle band
    if d <= t_good:
        mild_score = d / max(t_good, 1e-6)
    elif d <= t_mild:
        mild_score = 1.0 - (d - t_good) / max(t_mild - t_good, 1e-6)
    else:
        mild_score = 0.0

    # Severe score increases after t_mild
    if d <= t_mild:
        severe_score = 0.0
    else:
        severe_score = min(1.0, (d - t_mild) / max(t_mild, 1e-6))

    scores = {
        "good": good_score,
        "mild": mild_score,
        "severe": severe_score,
    }
    total = sum(scores.values())
    if total <= 0:
        return {"good": 1/3, "mild": 1/3, "severe": 1/3}

    for k in scores:
        scores[k] /= total

    return scores


# --- Feature extraction (same logic as training) ---
def extract_features_from_landmarks(landmarks, img_shape):
    """
    Use upper-body keypoints (nose, eyes, ears, shoulders),
    normalized by shoulder width and centered at shoulder midpoint.
    Returns (feature_vector, keypoints_for_drawing)
    """
    l_sh = landmarks[11]
    r_sh = landmarks[12]

    center_x = (l_sh.x + r_sh.x) / 2.0
    center_y = (l_sh.y + r_sh.y) / 2.0
    width = np.linalg.norm(
        np.array([l_sh.x, l_sh.y]) - np.array([r_sh.x, r_sh.y])
    )
    if width == 0:
        width = 1.0

    indices = [0, 2, 5, 7, 8, 11, 12]  # nose, eyes, ears, shoulders
    features = []

    h, w, _ = img_shape
    keypoints = {}

    for idx in indices:
        lm = landmarks[idx]
        norm_x = (lm.x - center_x) / width
        norm_y = (lm.y - center_y) / width
        features.extend([norm_x, norm_y])
        px, py = int(lm.x * w), int(lm.y * h)
        keypoints[idx] = (px, py)

    return features, keypoints


# --- Video Processor with Calibration + Distance-based Classification ---
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            model_complexity=1
        )

        # Personal standard posture (baseline)
        self.baseline = None
        self.calibrate_now = False  # set True when button pressed

        # Distance smoothing
        self.distance_history = deque(maxlen=10)

        # Shared outputs
        self.latest_probs = {"good": 0.0, "mild": 0.0, "severe": 0.0}
        self.latest_pred = "good"
        self.latest_distance = 0.0

        # Sound alert
        self.severe_consecutive_frames = 0
        self.trigger_sound = False

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            try:
                # 1. Feature extraction
                features, keypoints = extract_features_from_landmarks(
                    landmarks, img.shape
                )

                # 2. Calibration: when flag is True, save current features as baseline
                if self.calibrate_now:
                    self.baseline = np.array(features)
                    self.distance_history.clear()
                    self.calibrate_now = False

                # 3. If baseline exists, compute distance and map to probabilities
                if self.baseline is not None:
                    diff = np.array(features) - np.array(self.baseline)
                    dist = float(np.linalg.norm(diff))
                    self.distance_history.append(dist)
                    avg_dist = float(np.mean(self.distance_history))

                    self.latest_distance = avg_dist
                    prob_dict = distance_to_probs(avg_dist)
                    self.latest_probs = prob_dict
                    self.latest_pred = max(prob_dict, key=prob_dict.get)
                else:
                    # No baseline yet → assume good temporarily
                    self.latest_distance = 0.0
                    self.latest_probs = {"good": 1.0, "mild": 0.0, "severe": 0.0}
                    self.latest_pred = "good"

                current_pred = self.latest_pred

                # 4. Visualization color
                color = (0, 255, 0)  # green
                if current_pred == "mild":
                    color = (0, 255, 255)  # yellow
                elif current_pred == "severe":
                    color = (0, 0, 255)  # red

                # Draw keypoints
                for _, (px, py) in keypoints.items():
                    cv2.circle(img, (px, py), 5, color, -1)

                # Draw shoulders and neck line
                if 11 in keypoints and 12 in keypoints:
                    cv2.line(img, keypoints[11], keypoints[12], color, 2)
                if 0 in keypoints and 11 in keypoints and 12 in keypoints:
                    sh_center = (
                        (keypoints[11][0] + keypoints[12][0]) // 2,
                        (keypoints[11][1] + keypoints[12][1]) // 2,
                    )
                    cv2.line(img, sh_center, keypoints[0], color, 2)

                # 5. Sound trigger (if severe lasts for ~1 second)
                if current_pred == "severe":
                    self.severe_consecutive_frames += 1
                    if self.severe_consecutive_frames > 30:
                        self.trigger_sound = True
                else:
                    self.severe_consecutive_frames = 0
                    self.trigger_sound = False

            except Exception:
                pass

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# --- Layout: Left = Video, Right = Status ---
col_video, col_info = st.columns([3, 2])

ctx = None

# 왼쪽: 웹캠 + 캘리브레이션 버튼
with col_video:
    st.subheader("Webcam")
    ctx = webrtc_streamer(
        key="posture-pro",
        video_processor_factory=VideoProcessor,
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # 👉 영상 바로 아래 Calibration 버튼
    st.markdown("### Calibration")
    calib_msg_ph = st.empty()
    if st.button("📏 Set your best upright posture now!"):
        if ctx and ctx.video_processor:
            ctx.video_processor.calibrate_now = True
            calib_msg_ph.success(
                "✅ Your standard posture has been saved! "
                "Try to return to this position when you want a 'GOOD' rating."
            )
        else:
            calib_msg_ph.warning(
                "Webcam is not ready yet. Please wait a moment and try again."
            )

# 오른쪽: 라이브 상태 + 점수 바 + 거리
with col_info:
    st.subheader("📊 Live Status")

    status_ph = st.empty()
    advice_ph = st.empty()

    st.markdown("---")
    st.markdown("### Posture Scores")
    st.write("Good:")
    bar_good_ph = st.empty()
    st.write("Mild:")
    bar_mild_ph = st.empty()
    st.write("Severe:")
    bar_severe_ph = st.empty()

    st.markdown("---")
    dist_ph = st.empty()

    # Placeholders for sound & voice JS calls
    sound_ph = st.empty()
    tts_ph = st.empty()


# ✅ NEW: 사진 업로드용 섹션
st.markdown("---")
st.subheader("📷 Photo Upload Diagnosis (beta)")
st.markdown(
    "Upload a photo to get a posture diagnosis. "
    "For best results, **first set your standard posture via webcam or a good photo.**"
)

# 세션 상태에 baseline 저장할 공간
if "baseline_vector" not in st.session_state:
    st.session_state["baseline_vector"] = None

uploaded_file = st.file_uploader(
    "Upload a photo (front / side view where shoulders and head are visible)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # 파일 → OpenCV 이미지(BGR)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        st.error("❌ Could not read the image. Please try another file.")
    else:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Mediapipe로 포즈 추출
        results_img = pose_static.process(img_rgb)

        if not results_img.pose_landmarks:
            st.error("😥 Could not detect your body in this photo. Try a clearer upper-body photo.")
        else:
            landmarks_img = results_img.pose_landmarks.landmark
            features_img, keypoints_img = extract_features_from_landmarks(
                landmarks_img, img_bgr.shape
            )

            # 두 컬럼: 왼쪽은 사진, 오른쪽은 결과
            col_img, col_res = st.columns([3, 2])

            with col_img:
                # 사진 위에 keypoint + 라인 그림
                color = (0, 255, 0)  # 기본 green
                # 어깨/목 라인 먼저 (나중에 색 업데이트될 수 있어서 한번 더 그림)
                if 11 in keypoints_img and 12 in keypoints_img:
                    cv2.line(img_bgr, keypoints_img[11], keypoints_img[12], color, 2)
                if 0 in keypoints_img and 11 in keypoints_img and 12 in keypoints_img:
                    sh_center = (
                        (keypoints_img[11][0] + keypoints_img[12][0]) // 2,
                        (keypoints_img[11][1] + keypoints_img[12][1]) // 2,
                    )
                    cv2.line(img_bgr, sh_center, keypoints_img[0], color, 2)

                for _, (px, py) in keypoints_img.items():
                    cv2.circle(img_bgr, (px, py), 5, color, -1)

                st.image(
                    cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                    caption="Uploaded Photo with Keypoints",
                    use_column_width=True,
                )

            with col_res:
                st.markdown("### Result for This Photo")

                # 버튼 2개: 이 사진으로 baseline 설정 / 기존 baseline으로 분석
                set_baseline = st.button("✅ Use this photo as my standard posture")
                analyze_photo = st.button("📊 Analyze posture vs my standard", key="analyze_photo_btn")

                if set_baseline:
                    st.session_state["baseline_vector"] = features_img
                    st.success(
                        "Saved this photo as your **standard posture**. "
                        "Future photo / webcam analysis will compare to this."
                    )

                    # 웹캠 processor에도 baseline을 동기화 (가능하면)
                    if ctx and ctx.video_processor:
                        ctx.video_processor.baseline = np.array(features_img)

                if analyze_photo:
                    baseline_vec = st.session_state.get("baseline_vector", None)

                    if baseline_vec is None:
                        st.warning(
                            "No standard posture saved yet. "
                            "First, calibrate with the webcam or press "
                            "'Use this photo as my standard posture'."
                        )
                    else:
                        diff = np.array(features_img) - np.array(baseline_vec)
                        dist_img = float(np.linalg.norm(diff))
                        prob_dict_img = distance_to_probs(dist_img)
                        pred_img = max(prob_dict_img, key=prob_dict_img.get)

                        # 상태 텍스트
                        if pred_img == "good":
                            st.markdown(
                                "<div class='good-text'>GOOD 😊</div>",
                                unsafe_allow_html=True,
                            )
                            st.markdown(
                                "<div class='advice-box'>✅ This photo is close to your standard posture.</div>",
                                unsafe_allow_html=True,
                            )
                        elif pred_img == "mild":
                            st.markdown(
                                "<div class='mild-text'>MILD 😐</div>",
                                unsafe_allow_html=True,
                            )
                            st.markdown(
                                "<div class='advice-box'>💡 Slight forward head or shoulder tension detected.</div>",
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                "<div class='severe-text'>SEVERE 🐢</div>",
                                unsafe_allow_html=True,
                            )
                            st.markdown(
                                "<div class='advice-box'>🚨 Pull your chin back and open your chest more.</div>",
                                unsafe_allow_html=True,
                            )

                        # 점수 바
                        g_img = prob_dict_img.get("good", 0.0) * 100
                        m_img = prob_dict_img.get("mild", 0.0) * 100
                        s_img = prob_dict_img.get("severe", 0.0) * 100

                        st.write("Good:")
                        st.progress(int(g_img), text=f"{g_img:.1f}%")
                        st.write("Mild:")
                        st.progress(int(m_img), text=f"{m_img:.1f}%")
                        st.write("Severe:")
                        st.progress(int(s_img), text=f"{s_img:.1f}%")

                        st.markdown(
                            f"Deviation from your standard posture in this photo: **{dist_img:.3f}**"
                        )


# --- Main Update Loop (Webcam) ---
if ctx and ctx.state.playing:
    while True:
        if not ctx.state.playing:
            break

        vp = ctx.video_processor
        if vp is not None:
            # ✅ NEW: 웹캠에서 잡은 baseline을 세션에도 공유해서 사진에서 재사용
            if vp.baseline is not None:
                st.session_state["baseline_vector"] = list(vp.baseline)

            probs = vp.latest_probs
            pred = vp.latest_pred
            trigger_sound = vp.trigger_sound
            dist = vp.latest_distance

            # Status & advice + voice
            if pred == "good":
                status_ph.markdown(
                    "<div class='good-text'>GOOD 😊</div>", unsafe_allow_html=True
                )
                advice_ph.markdown(
                    "<div class='advice-box'>✅ Perfect alignment! Keep this posture.</div>",
                    unsafe_allow_html=True,
                )
                tts_ph.markdown(
                    "<script>updatePostureStatus('GOOD');</script>",
                    unsafe_allow_html=True,
                )

            elif pred == "mild":
                status_ph.markdown(
                    "<div class='mild-text'>MILD 😐</div>", unsafe_allow_html=True
                )
                advice_ph.markdown(
                    "<div class='advice-box'>💡 Lift your head slightly and relax your shoulders.</div>",
                    unsafe_allow_html=True,
                )
                tts_ph.markdown(
                    "<script>updatePostureStatus('MILD');</script>",
                    unsafe_allow_html=True,
                )

            else:
                status_ph.markdown(
                    "<div class='severe-text'>SEVERE 🐢</div>", unsafe_allow_html=True
                )
                advice_ph.markdown(
                    "<div class='advice-box'>🚨 <b>Pull your chin back</b> and open your chest.</div>",
                    unsafe_allow_html=True,
                )
                tts_ph.markdown(
                    "<script>updatePostureStatus('SEVERE');</script>",
                    unsafe_allow_html=True,
                )

            # 세 개의 가로 Progress bar (Good / Mild / Severe)
            g = probs.get("good", 0.0) * 100
            m = probs.get("mild", 0.0) * 100
            s = probs.get("severe", 0.0) * 100

            bar_good_ph.progress(int(g), text=f"{g:.1f}%")
            bar_mild_ph.progress(int(m), text=f"{m:.1f}%")
            bar_severe_ph.progress(int(s), text=f"{s:.1f}%")

            # Baseline에서의 거리 표시
            dist_ph.markdown(
                f"Current deviation from your standard posture: **{dist:.3f}**"
            )

            # Beep alert only for SEVERE (when trigger flag is set)
            if trigger_sound:
                sound_ph.markdown(
                    "<script>playAlert();</script>", unsafe_allow_html=True
                )
            else:
                sound_ph.empty()

        time.sleep(0.1)
