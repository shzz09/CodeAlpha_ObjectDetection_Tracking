
import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO
from collections import Counter

# PAGE CONFIG
st.set_page_config(
    page_title="VisionTrack AI",
    page_icon="🎯",
    layout="wide"
)

# CUSTOM CSS - PROFESSIONAL UI
st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #0F172A;
    color: #E2E8F0;
}

.block-container {
    max-width: 1400px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Header */
h1, h2, h3, h4 {
    color: #F1F5F9 !important;
    font-weight: 650;
    letter-spacing: -0.4px;
}

p, label {
    color: #CBD5E1;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111C30;
    border-right: 1px solid #29384F;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #F1F5F9 !important;
}

/* Sidebar branding */
.sidebar-brand {
    font-size: 27px;
    font-weight: 800;
    letter-spacing: -1px;
    color: #F8FAFC;
}

.sidebar-brand span {
    color: #8EADD8;
}

.sidebar-subtitle {
    font-size: 12px;
    color: #91A2BA;
    margin-top: 5px;
}

/* Main title */
.main-brand {
    font-size: 37px;
    font-weight: 800;
    letter-spacing: -1.5px;
    color: #F8FAFC;
}

.main-brand span {
    color: #8EADD8;
}

.subtitle {
    color: #94A3B8;
    font-size: 14px;
    margin-top: 5px;
}

/* Section heading */
.section-subtitle {
    color: #94A3B8;
    font-size: 14px;
    margin-bottom: 18px;
}

/* Metrics */
[data-testid="stMetric"] {
    background: #18263B;
    border: 1px solid #30435D;
    border-radius: 12px;
    padding: 21px;
}

[data-testid="stMetricLabel"] {
    color: #AABBD0 !important;
    font-size: 13px;
}

[data-testid="stMetricValue"] {
    color: #A4BDE0 !important;
    font-weight: 750;
}

/* Upload section */
[data-testid="stFileUploader"] {
    background: #142238;
    border: 1px dashed #526D91;
    border-radius: 12px;
    padding: 15px;
}

[data-testid="stFileUploader"] section {
    background: transparent;
}

/* Buttons */
.stButton > button,
.stDownloadButton > button {
    background: #527DAF;
    color: #FFFFFF !important;
    border: 1px solid #527DAF;
    border-radius: 8px;
    min-height: 43px;
    font-weight: 600;
    transition: 0.2s ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background: #416B9B;
    border-color: #416B9B;
    color: #FFFFFF !important;
}

/* Slider and select */
div[data-baseweb="select"] > div {
    background: #18263B;
    border-color: #30435D;
    border-radius: 8px;
}

/* Result category rows */
.category-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #18263B;
    border: 1px solid #30435D;
    border-radius: 9px;
    padding: 14px 17px;
    margin-bottom: 10px;
    color: #E2E8F0;
    font-size: 14px;
}

.category-count {
    color: #A4BDE0;
    font-weight: 700;
}

/* Divider */
hr {
    border-color: #29384F;
}

/* Image and video */
[data-testid="stImage"] img,
[data-testid="stVideo"] video {
    border-radius: 11px;
}

/* Alerts */
div[data-testid="stAlert"] {
    border-radius: 10px;
}

/* Footer */
.footer {
    text-align: center;
    color: #8493A9;
    font-size: 12px;
    padding: 18px;
    line-height: 1.8;
}

/* Status */
.status {
    display: inline-block;
    background: #1B302F;
    border: 1px solid #34534C;
    color: #9BD2BC;
    padding: 7px 13px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 650;
}

</style>
""", unsafe_allow_html=True)


# LOAD YOLO MODEL
@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")

model = load_model()


# HEADER
st.markdown("""
<div class="main-brand">
    VisionTrack <span>AI</span>
</div>
<div class="subtitle">
    Real-Time Object Detection & Tracking System
</div>
""", unsafe_allow_html=True)

st.write("")

st.divider()


# SIDEBAR
st.sidebar.markdown("""
<div class="sidebar-brand">
    Vision<span>Track</span>
</div>
<div class="sidebar-subtitle">
    Intelligent Vision Platform
</div>
""", unsafe_allow_html=True)

st.sidebar.write("")
st.sidebar.divider()

st.sidebar.markdown("### Control Panel")

mode = st.sidebar.radio(
    "Select Detection Mode",
    ["Upload Video", "Image Detection"]
)

st.sidebar.write("")

st.sidebar.markdown("### Detection Settings")

confidence = st.sidebar.slider(
    "Detection Confidence",
    min_value=0.10,
    max_value=0.90,
    value=0.35,
    step=0.05
)

st.sidebar.caption(
    "Adjust the minimum confidence level for object detection."
)

st.sidebar.divider()

st.sidebar.markdown(
    '<span class="status">● SYSTEM ONLINE</span>',
    unsafe_allow_html=True
)


# DASHBOARD
st.markdown("## Detection Dashboard")

st.markdown(
    '<div class="section-subtitle">'
    'Upload images or videos to analyze and track objects.'
    '</div>',
    unsafe_allow_html=True
)


# IMAGE DETECTION
if mode == "Image Detection":

    st.markdown("### Image Detection")

    st.markdown(
        '<div class="section-subtitle">'
        'Upload an image to identify and classify objects automatically.'
        '</div>',
        unsafe_allow_html=True
    )

    uploaded_image = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image:

        file_bytes = uploaded_image.getvalue()
        image_array = cv2.imdecode(
            __import__("numpy").frombuffer(
                file_bytes, dtype=__import__("numpy").uint8
            ),
            cv2.IMREAD_COLOR
        )

        if st.button("Detect Objects", use_container_width=True):

            results = model.predict(
                image_array,
                conf=confidence,
                verbose=False
            )

            result = results[0]
            annotated = result.plot()

            st.success("Image detection completed successfully!")

            st.markdown("### Detection Results")

            left, right = st.columns([3, 2])

            boxes = result.boxes

            names = [
                model.names[int(box.cls[0].item())]
                for box in boxes
            ]

            with left:

                st.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    caption="Object Detection Result",
                    use_container_width=True
                )

            with right:

                col1, col2 = st.columns(2)

                col1.metric("Total Objects", len(boxes))

                col2.metric(
                    "Object Categories",
                    len(set(names))
                )

                st.markdown("### Detected Objects")

                for name, count in Counter(names).items():

                    st.markdown(
                        f"""
                        <div class="category-row">
                            <span>{name.capitalize()}</span>
                            <span class="category-count">{count}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


# VIDEO DETECTION
elif mode == "Upload Video":

    st.markdown("### Video Object Detection & Tracking")

    st.markdown(
        '<div class="section-subtitle">'
        'Upload a video to detect objects and track their movement '
        'across multiple frames.'
        '</div>',
        unsafe_allow_html=True
    )

    uploaded_video = st.file_uploader(
        "Choose a video",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_video:

        st.video(uploaded_video)

        if st.button(
            "Start Object Detection",
            use_container_width=True
        ):

            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".mp4"
            ) as temp:

                temp.write(uploaded_video.getvalue())
                video_path = temp.name

            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():

                st.error("Unable to open video.")
                os.remove(video_path)

            else:

                total_frames = int(
                    cap.get(cv2.CAP_PROP_FRAME_COUNT)
                )

                fps = cap.get(cv2.CAP_PROP_FPS)

                if fps <= 0:
                    fps = 25

                width = int(
                    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                )

                height = int(
                    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                )

                output_path = os.path.join(
                    "outputs", "processed_video.mp4"
                )

                os.makedirs("outputs", exist_ok=True)

                writer = cv2.VideoWriter(
                    output_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height)
                )

                st.markdown("### Processing Video")

                display = st.empty()
                progress = st.progress(0)

                frame_count = 0
                total_detections = 0
                tracking_ids = set()
                object_counts = Counter()

                while True:

                    ret, frame = cap.read()

                    if not ret:
                        break

                    results = model.track(
                        frame,
                        persist=True,
                        tracker="bytetrack.yaml",
                        conf=confidence,
                        verbose=False
                    )

                    result = results[0]
                    annotated = result.plot()

                    if result.boxes is not None:

                        total_detections += len(
                            result.boxes
                        )

                        for box in result.boxes:

                            class_id = int(
                                box.cls[0].item()
                            )

                            object_name = model.names[class_id]

                            object_counts[object_name] += 1

                            if box.id is not None:

                                tracking_ids.add(
                                    int(box.id[0].item())
                                )

                    writer.write(annotated)

                    display.image(
                        cv2.cvtColor(
                            annotated, cv2.COLOR_BGR2RGB
                        ),
                        caption=f"Processing frame {frame_count + 1}",
                        use_container_width=True
                    )

                    frame_count += 1

                    if total_frames > 0:

                        progress.progress(
                            min(frame_count / total_frames, 1.0)
                        )

                cap.release()
                writer.release()

                st.success("Detection completed successfully!")

                st.markdown("## Detection Results")

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Frames Processed",
                    frame_count
                )

                c2.metric(
                    "Total Detections",
                    total_detections
                )

                c3.metric(
                    "Unique Tracking IDs",
                    len(tracking_ids)
                )

                st.markdown("### Detected Object Categories")

                for name, count in object_counts.most_common():

                    st.markdown(
                        f"""
                        <div class="category-row">
                            <span>{name.capitalize()}</span>
                            <span class="category-count">
                                {count:,} detections across frames
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                if os.path.exists(output_path):

                    with open(output_path, "rb") as file:

                        st.download_button(
                            "Download Processed Video",
                            data=file,
                            file_name="VisionTrack_Processed.mp4",
                            mime="video/mp4",
                            use_container_width=True
                        )

                os.remove(video_path)


# FOOTER
st.divider()

st.markdown("""
<div class="footer">
    <b>VISIONTRACK AI</b><br>
    Real-Time Object Detection & Tracking System<br>
    Powered by YOLO + OpenCV + ByteTrack
</div>
""", unsafe_allow_html=True)

