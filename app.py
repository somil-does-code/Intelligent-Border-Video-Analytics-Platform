from flask import Flask, render_template, jsonify, Response, request
import cv2
import time
import random
import threading
from datetime import datetime
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# ============================================================
# CONFIGURATION
# ============================================================

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {
    "mp4",
    "mov",
    "avi",
    "mkv",
    "webm"
}

# Process every Nth frame.
# 1 = every frame
# 2 = every second frame (faster)
FRAME_SKIP = 2

# Maximum number of alerts stored
MAX_ALERTS = 8


# ============================================================
# GLOBAL VIDEO STATE
# ============================================================

latest_uploaded_video = None

video_lock = threading.Lock()


# ============================================================
# AI ENGINE
# ============================================================

ai_engine = None
ai_lock = threading.Lock()


# ============================================================
# CAMERAS
# ============================================================

CAMERAS = [
    {
        "id": "A3",
        "name": "Sector A-3",
        "type": "Thermal Camera",
        "status": "ONLINE"
    },
    {
        "id": "B1",
        "name": "Sector B-1",
        "type": "Fence Perimeter",
        "status": "ONLINE"
    },
    {
        "id": "C4",
        "name": "Sector C-4",
        "type": "Checkpoint",
        "status": "ONLINE"
    }
]


# ============================================================
# ALERTS
# ============================================================

alert_types = [
    "Perimeter Intrusion",
    "Thermal Anomaly",
    "Vehicle Trespassing",
    "Unidentified Target",
    "Suspicious Activity"
]

sectors = [
    "Sector A-3",
    "Sector B-1",
    "Sector C-4"
]

alerts = [
    {
        "type": "Perimeter Intrusion",
        "sector": "Sector A-3",
        "details": "Target spotted near the fence",
        "confidence": 94.2,
        "time": "10:42:15 AM"
    }
]


# ============================================================
# LIVE METRICS
# ============================================================

metrics = {
    "fps": 30,
    "humans": 1,
    "vehicles": 0,
    "tracks": 1,
    "latency": 14,
    "bandwidth": 85
}


# ============================================================
# UTILITY
# ============================================================

def now_time():
    return datetime.now().strftime("%I:%M:%S %p")


def allowed_file(filename):
    if not filename:
        return False

    if "." not in filename:
        return False

    extension = filename.rsplit(".", 1)[1].lower()

    return extension in ALLOWED_EXTENSIONS


# ============================================================
# ALERT CREATION
# ============================================================

def add_alert(
    alert_type,
    sector,
    details,
    confidence=None
):

    item = {
        "type": alert_type,
        "sector": sector,
        "details": details,
        "confidence": confidence,
        "time": now_time()
    }

    alerts.insert(0, item)

    del alerts[MAX_ALERTS:]

    return item


# ============================================================
# LOAD AI DETECTOR
# ============================================================

def load_ai():

    global ai_engine

    with ai_lock:

        if ai_engine is not None:
            return ai_engine

        try:

            from detection.human_vehicle_detector import (
                HumanVehicleDetector
            )

            detector = HumanVehicleDetector()

            ai_engine = {
                "detector": detector
            }

            print("----------------------------------------")
            print("[AI] HumanVehicleDetector loaded")
            print("[AI] YOLO tracking is enabled")
            print("----------------------------------------")

            return ai_engine

        except Exception as exc:

            print("----------------------------------------")
            print("[AI ERROR] Could not load detector")
            print("[AI ERROR]", exc)
            print("----------------------------------------")

            return None


# ============================================================
# DEMO FRAME
# ============================================================

def make_demo_frame(camera_id="A3"):

    import numpy as np

    width = 960
    height = 540

    frame = np.zeros(
        (height, width, 3),
        dtype="uint8"
    )

    frame[:] = (8, 12, 22)

    # Grid
    for x in range(0, width, 40):

        cv2.line(
            frame,
            (x, 0),
            (x, height),
            (22, 31, 46),
            1
        )

    for y in range(0, height, 40):

        cv2.line(
            frame,
            (0, y),
            (width, y),
            (22, 31, 46),
            1
        )

    cam = next(
        (
            c for c in CAMERAS
            if c["id"] == camera_id
        ),
        CAMERAS[0]
    )

    # Header
    cv2.putText(
        frame,
        f"LIVE | {cam['name']} ({cam['type']})",
        (24, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (230, 235, 245),
        2
    )

    cv2.putText(
        frame,
        "IBVAP AI ANALYTICS  •  PROTOTYPE MODE",
        (24, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (70, 220, 170),
        1
    )

    # Demo person
    bx = 390
    by = 150
    bw = 150
    bh = 230

    cv2.rectangle(
        frame,
        (bx, by),
        (bx + bw, by + bh),
        (40, 90, 245),
        2
    )

    cv2.rectangle(
        frame,
        (bx, by - 25),
        (bx + 165, by),
        (40, 90, 245),
        -1
    )

    cv2.putText(
        frame,
        "PERSON 94.2% ID:17",
        (bx + 6, by - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (255, 255, 255),
        1
    )

    # Virtual fence
    cv2.line(
        frame,
        (80, 410),
        (880, 410),
        (50, 70, 245),
        2
    )

    cv2.putText(
        frame,
        "VIRTUAL FENCE",
        (410, 398),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (80, 100, 250),
        1
    )

    # Bottom metrics
    cv2.rectangle(
        frame,
        (20, height - 48),
        (width - 20, height - 18),
        (4, 8, 16),
        -1
    )

    cv2.putText(
        frame,
        f"HUMANS: {metrics['humans']}   "
        f"VEHICLES: {metrics['vehicles']}   "
        f"TRACKS: {metrics['tracks']}   "
        f"FPS: {metrics['fps']}",
        (30, height - 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (210, 220, 235),
        1
    )

    return frame


# ============================================================
# ENCODE FRAME
# ============================================================

def encode_frame(frame):

    success, jpg = cv2.imencode(
        ".jpg",
        frame,
        [
            int(cv2.IMWRITE_JPEG_QUALITY),
            82
        ]
    )

    if not success:
        return None

    return (
        b"--frame\r\n"
        b"Content-Type: image/jpeg\r\n\r\n"
        + jpg.tobytes()
        + b"\r\n"
    )


# ============================================================
# DEMO STREAM
# ============================================================

def generate_demo_stream(camera_id):

    while True:

        frame = make_demo_frame(camera_id)

        data = encode_frame(frame)

        if data:
            yield data

        time.sleep(1 / 15)


# ============================================================
# RAW VIDEO STREAM
# ============================================================

def generate_raw_video_stream(
    video_path,
    camera_id
):

    print("[VIDEO] Opening:", video_path)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():

        print("[VIDEO ERROR] Cannot open video")

        yield from generate_demo_stream(camera_id)

        return

    source_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if not source_fps or source_fps <= 0:
        source_fps = 25

    # Don't intentionally make video slower
    delay = 1 / min(source_fps, 30)

    while True:

        success, frame = cap.read()

        if not success:

            # Restart video
            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                0
            )

            success, frame = cap.read()

            if not success:
                break

        # Resize only if video is extremely large
        height, width = frame.shape[:2]

        if width > 1280:

            scale = 1280 / width

            frame = cv2.resize(
                frame,
                (
                    int(width * scale),
                    int(height * scale)
                )
            )

        cv2.putText(
            frame,
            "IBVAP AI ANALYTICS",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (70, 220, 170),
            2
        )

        cv2.putText(
            frame,
            "UPLOADED VIDEO",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1
        )

        data = encode_frame(frame)

        if data:
            yield data

        time.sleep(delay)

    cap.release()


# ============================================================
# AI VIDEO STREAM
# ============================================================

def generate_ai_stream(
    video_path,
    camera_id
):

    engine = load_ai()

    # --------------------------------------------------------
    # If detector could not load
    # --------------------------------------------------------

    if engine is None:

        print(
            "[AI] Detector unavailable."
        )

        # Still show the uploaded video
        if (
            video_path
            and os.path.exists(video_path)
        ):

            yield from generate_raw_video_stream(
                video_path,
                camera_id
            )

        else:

            yield from generate_demo_stream(
                camera_id
            )

        return

    # --------------------------------------------------------
    # Open uploaded video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():

        print(
            "[VIDEO ERROR] "
            "Could not open uploaded video."
        )

        yield from generate_demo_stream(
            camera_id
        )

        return

    source_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if not source_fps or source_fps <= 0:
        source_fps = 25

    frame_number = 0

    last_tracked_objects = []

    last_process_time = time.time()

    # --------------------------------------------------------
    # VIDEO LOOP
    # --------------------------------------------------------

    while True:

        success, frame = cap.read()

        # ----------------------------------------------------
        # Restart when video ends
        # ----------------------------------------------------

        if not success:

            print(
                "[VIDEO] Restarting uploaded video..."
            )

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                0
            )

            # Reset detector tracking state
            try:

                detector = engine["detector"]

                if hasattr(
                    detector.model,
                    "predictor"
                ):

                    detector.model.predictor = None

            except Exception:
                pass

            frame_number = 0

            success, frame = cap.read()

            if not success:
                break

        frame_number += 1

        # ----------------------------------------------------
        # Resize very large videos
        # ----------------------------------------------------

        height, width = frame.shape[:2]

        if width > 1280:

            scale = 1280 / width

            frame = cv2.resize(
                frame,
                (
                    int(width * scale),
                    int(height * scale)
                )
            )

        # ----------------------------------------------------
        # YOLO DETECTION
        # ----------------------------------------------------

        should_process = (
            frame_number % FRAME_SKIP == 0
        )

        if should_process:

            try:

                detector = engine["detector"]

                tracked_objects = (
                    detector.process_frame(frame)
                )

                last_tracked_objects = (
                    tracked_objects
                )

                # --------------------------------------------
                # Count humans
                # --------------------------------------------

                humans = sum(
                    obj.category == "human"
                    for obj in tracked_objects
                )

                # --------------------------------------------
                # Count vehicles
                # --------------------------------------------

                vehicles = sum(
                    obj.category == "vehicle"
                    for obj in tracked_objects
                )

                # --------------------------------------------
                # Count tracks
                # --------------------------------------------

                tracks = len(
                    tracked_objects
                )

                metrics.update({

                    "humans": humans,

                    "vehicles": vehicles,

                    "tracks": tracks,

                    "fps": int(
                        source_fps
                    )

                })

                # --------------------------------------------
                # Print useful debugging info
                # --------------------------------------------

                if frame_number % 30 == 0:

                    print(
                        f"[YOLO] "
                        f"Frame={frame_number} "
                        f"Humans={humans} "
                        f"Vehicles={vehicles} "
                        f"Tracks={tracks}"
                    )

            except Exception as exc:

                print(
                    "[YOLO FRAME ERROR]",
                    exc
                )

        # ----------------------------------------------------
        # DRAW DETECTIONS
        # ----------------------------------------------------

        try:

            detector = engine["detector"]

            annotated = (
                detector.annotate_frame(
                    frame,
                    last_tracked_objects
                )
            )

        except Exception as exc:

            print(
                "[ANNOTATION ERROR]",
                exc
            )

            annotated = frame.copy()

        # ----------------------------------------------------
        # IBVAP INFORMATION
        # ----------------------------------------------------

        cv2.putText(
            annotated,
            "IBVAP AI ANALYTICS",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (70, 220, 170),
            2
        )

        cv2.putText(
            annotated,
            "YOLO HUMAN / VEHICLE DETECTION",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        # ----------------------------------------------------
        # Bottom metrics
        # ----------------------------------------------------

        height, width = annotated.shape[:2]

        cv2.rectangle(
            annotated,
            (15, height - 50),
            (width - 15, height - 15),
            (5, 10, 20),
            -1
        )

        cv2.putText(
            annotated,
            f"HUMANS: {metrics['humans']}   "
            f"VEHICLES: {metrics['vehicles']}   "
            f"TRACKS: {metrics['tracks']}",
            (25, height - 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 230, 240),
            1
        )

        # ----------------------------------------------------
        # Encode
        # ----------------------------------------------------

        data = encode_frame(
            annotated
        )

        if data:
            yield data

        # ----------------------------------------------------
        # DON'T ADD AN EXTRA .03 SECOND DELAY
        # ----------------------------------------------------
        #
        # YOLO processing itself controls the speed.
        # Adding another sleep here was making the feed slower.
        #

    cap.release()


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# ============================================================
# CAMERA API
# ============================================================

@app.get("/api/cameras")
def api_cameras():

    return jsonify(
        CAMERAS
    )


# ============================================================
# DASHBOARD API
# ============================================================

@app.get("/api/dashboard")
def api_dashboard():

    return jsonify({

        "system": {

            "feeds": len(
                CAMERAS
            ),

            "latency": metrics[
                "latency"
            ],

            "mode": "prototype"

        },

        "metrics": metrics,

        "alerts": alerts,

        "timestamp":
            datetime.now().isoformat(
                timespec="seconds"
            )

    })


# ============================================================
# DEMO ALERT API
# ============================================================

@app.post("/api/demo-alert")
def api_demo_alert():

    if request.is_json:

        camera = request.json.get(
            "sector",
            "Sector A-3"
        )

    else:

        camera = "Sector A-3"

    item = add_alert(

        random.choice(
            alert_types
        ),

        camera,

        "AI event generated for demonstration",

        round(
            random.uniform(
                82,
                98
            ),
            1
        )

    )

    return jsonify(
        item
    )


# ============================================================
# UPLOAD VIDEO API
# ============================================================

@app.post("/api/upload-video")
def upload_video():

    global latest_uploaded_video

    # --------------------------------------------------------
    # Check request
    # --------------------------------------------------------

    if "video" not in request.files:

        return jsonify({

            "error":
                "No video file was uploaded."

        }), 400

    file = request.files[
        "video"
    ]

    # --------------------------------------------------------
    # Check filename
    # --------------------------------------------------------

    if not file.filename:

        return jsonify({

            "error":
                "No video file selected."

        }), 400

    filename = secure_filename(
        file.filename
    )

    # --------------------------------------------------------
    # Check extension
    # --------------------------------------------------------

    if not allowed_file(
        filename
    ):

        return jsonify({

            "error":
                "Unsupported video format. "
                "Use MP4, MOV, AVI, MKV or WEBM."

        }), 400

    # --------------------------------------------------------
    # Create unique filename
    # --------------------------------------------------------

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    filename = (
        timestamp
        + "_"
        + filename
    )

    filepath = os.path.join(
        app.config[
            "UPLOAD_FOLDER"
        ],
        filename
    )

    # --------------------------------------------------------
    # Save video
    # --------------------------------------------------------

    try:

        file.save(
            filepath
        )

    except Exception as exc:

        print(
            "[UPLOAD ERROR]",
            exc
        )

        return jsonify({

            "error":
                "Could not save uploaded video."

        }), 500

    # --------------------------------------------------------
    # Verify video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        filepath
    )

    if not cap.isOpened():

        cap.release()

        try:
            os.remove(
                filepath
            )
        except Exception:
            pass

        return jsonify({

            "error":
                "Uploaded file is not a valid video."

        }), 400

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    cap.release()

    # --------------------------------------------------------
    # Store latest video
    # --------------------------------------------------------

    with video_lock:

        latest_uploaded_video = (
            filepath
        )

    print("----------------------------------------")
    print("[UPLOAD SUCCESS]")
    print("File:", filepath)
    print("Resolution:", width, "x", height)
    print("FPS:", fps)
    print("Frames:", frame_count)
    print("----------------------------------------")

    return jsonify({

        "success": True,

        "filename": filename,

        "width": width,

        "height": height,

        "fps": fps,

        "frames": frame_count,

        "message":
            "Video uploaded successfully."

    })


# ============================================================
# VIDEO FEED
# ============================================================

@app.get("/video_feed")
def video_feed():

    global latest_uploaded_video

    camera_id = request.args.get(
        "camera",
        "A3"
    )

    mode = request.args.get(
        "mode",
        "demo"
    )

    # --------------------------------------------------------
    # AI MODE
    # --------------------------------------------------------

    if mode == "ai":

        with video_lock:

            uploaded_video = (
                latest_uploaded_video
            )

        # ----------------------------------------------------
        # Use uploaded video
        # ----------------------------------------------------

        if (
            uploaded_video
            and os.path.exists(
                uploaded_video
            )
        ):

            generator = generate_ai_stream(

                uploaded_video,

                camera_id

            )

        else:

            # No uploaded video
            generator = generate_demo_stream(
                camera_id
            )

    # --------------------------------------------------------
    # DEMO MODE
    # --------------------------------------------------------

    else:

        generator = generate_demo_stream(
            camera_id
        )

    return Response(

        generator,

        mimetype=
            "multipart/x-mixed-replace; "
            "boundary=frame"

    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return jsonify({

        "status": "online",

        "ai_loaded":
            ai_engine is not None,

        "video_uploaded":
            latest_uploaded_video is not None

    })


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    app.run(

        host="0.0.0.0",

        port=int(
            os.environ.get(
                "PORT",
                5000
            )
        ),

        debug=False

    )
