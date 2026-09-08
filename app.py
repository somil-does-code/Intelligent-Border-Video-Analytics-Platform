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
# UPLOAD CONFIGURATION
# ============================================================

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Stores the most recently uploaded video
latest_uploaded_video = None

# ============================================================
# AI ENGINE
# ============================================================

ai_engine = None
ai_lock = threading.Lock()

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
    },
]

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


# ============================================================
# DEMO FRAME
# ============================================================

def make_demo_frame(camera_id="A3"):
    """
    Generate a demo CCTV frame so the website works
    even when no real camera/video is being used.
    """

    w, h = 960, 540

    import numpy as np

    frame = np.zeros((h, w, 3), dtype="uint8")
    frame[:] = (8, 12, 22)

    # Grid
    for x in range(0, w, 40):
        cv2.line(
            frame,
            (x, 0),
            (x, h),
            (22, 31, 46),
            1
        )

    for y in range(0, h, 40):
        cv2.line(
            frame,
            (0, y),
            (w, y),
            (22, 31, 46),
            1
        )

    cam = next(
        (c for c in CAMERAS if c["id"] == camera_id),
        CAMERAS[0]
    )

    cv2.putText(
        frame,
        f"LIVE  |  {cam['name']} ({cam['type']})",
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

    # Demo target
    bx, by, bw, bh = 390, 150, 150, 230

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
        "PERSON  94.2%  ID:17",
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

    # Bottom information bar
    cv2.rectangle(
        frame,
        (20, h - 48),
        (w - 20, h - 18),
        (4, 8, 16),
        -1
    )

    cv2.putText(
        frame,
        f"HUMANS: {metrics['humans']}   "
        f"VEHICLES: {metrics['vehicles']}   "
        f"TRACKS: {metrics['tracks']}   "
        f"FPS: {metrics['fps']}",
        (30, h - 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (210, 220, 235),
        1
    )

    return frame


# ============================================================
# AI LOADER
# ============================================================

def load_ai():
    global ai_engine

    with ai_lock:

        if ai_engine is not None:
            return ai_engine

        try:

            from detection.human_vehicle_detector import HumanVehicleDetector
            from anpr.anpr_pipeline import ANPRPipeline
            from suspicious.pose_detector import SuspiciousPoseDetector

            detector = HumanVehicleDetector()

            ai_engine = {
                "detector": detector,
                "anpr": None,
                "suspicious": SuspiciousPoseDetector()
            }

            ai_engine["anpr"] = ANPRPipeline(detector)

            print("[AI] AI engine loaded successfully")

            return ai_engine

        except Exception as exc:

            print("[AI MODE ERROR]", exc)

            return None


# ============================================================
# JPEG STREAM HELPER
# ============================================================

def encode_frame(frame):

    ok, jpg = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), 82]
    )

    if not ok:
        return None

    return (
        b"--frame\r\n"
        b"Content-Type: image/jpeg\r\n\r\n"
        + jpg.tobytes()
        + b"\r\n"
    )


# ============================================================
# DEMO VIDEO STREAM
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

def generate_raw_video_stream(video_path, camera_id):

    print("[VIDEO] Opening uploaded video:", video_path)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():

        print("[VIDEO ERROR] Could not open uploaded video")

        yield from generate_demo_stream(camera_id)

        return

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 0:
        fps = 25

    delay = 1 / min(fps, 30)

    while True:

        ok, frame = cap.read()

        if not ok:

            # Restart video when it reaches the end
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            ok, frame = cap.read()

            if not ok:
                break

        # Add small IBVAP label
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

def generate_ai_stream(source, camera_id):

    engine = load_ai()

    # --------------------------------------------------------
    # If AI modules are unavailable, still show uploaded video
    # instead of falling back to the fake demo screen.
    # --------------------------------------------------------

    if engine is None:

        print("[AI] AI modules unavailable.")

        if source and os.path.exists(str(source)):

            print("[AI] Showing uploaded video without AI processing.")

            yield from generate_raw_video_stream(
                source,
                camera_id
            )

        else:

            yield from generate_demo_stream(camera_id)

        return

    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

    try:

        if str(source).isdigit():
            cap = cv2.VideoCapture(int(source))
        else:
            cap = cv2.VideoCapture(source)

    except Exception as exc:

        print("[VIDEO OPEN ERROR]", exc)

        yield from generate_demo_stream(camera_id)

        return

    if not cap.isOpened():

        print("[VIDEO ERROR] Video source could not be opened.")

        if source and os.path.exists(str(source)):
            yield from generate_raw_video_stream(
                source,
                camera_id
            )
        else:
            yield from generate_demo_stream(camera_id)

        return

    # --------------------------------------------------------
    # Processing loop
    # --------------------------------------------------------

    frame_count = 0

    fps_value = cap.get(cv2.CAP_PROP_FPS)

    if not fps_value or fps_value <= 0:
        fps_value = 25

    while True:

        ok, frame = cap.read()

        if not ok:

            # Restart video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            ok, frame = cap.read()

            if not ok:
                break

        frame_count += 1

        try:

            # ------------------------------------------------
            # HUMAN / VEHICLE DETECTION
            # ------------------------------------------------

            tracked = engine["detector"].process_frame(frame)

            # ------------------------------------------------
            # ANPR
            # ------------------------------------------------

            engine["anpr"].process_frame(
                frame,
                tracked
            )

            # ------------------------------------------------
            # SUSPICIOUS ACTIVITY
            # ------------------------------------------------

            pose_alert = engine["suspicious"].process_frame(frame)

            # ------------------------------------------------
            # DRAW DETECTIONS
            # ------------------------------------------------

            annotated = engine["detector"].annotate_frame(
                frame,
                tracked
            )

            # ------------------------------------------------
            # METRICS
            # ------------------------------------------------

            humans = sum(
                o.category == "human"
                for o in tracked
            )

            vehicles = sum(
                o.category == "vehicle"
                for o in tracked
            )

            metrics.update({
                "humans": humans,
                "vehicles": vehicles,
                "tracks": len(tracked),
                "fps": int(fps_value)
            })

            # ------------------------------------------------
            # SUSPICIOUS ACTIVITY ALERT
            # ------------------------------------------------

            if pose_alert:

                camera_name = next(
                    (
                        c["name"]
                        for c in CAMERAS
                        if c["id"] == camera_id
                    ),
                    camera_id
                )

                add_alert(
                    "Suspicious Activity",
                    camera_name,
                    "Pose anomaly detected",
                    88.0
                )

                cv2.rectangle(
                    annotated,
                    (10, 10),
                    (430, 55),
                    (0, 0, 255),
                    -1
                )

                cv2.putText(
                    annotated,
                    "ALERT: SUSPICIOUS ACTIVITY",
                    (20, 42),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2
                )

            # ------------------------------------------------
            # IBVAP LABEL
            # ------------------------------------------------

            cv2.putText(
                annotated,
                "IBVAP AI ANALYTICS",
                (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (70, 220, 170),
                2
            )

        except Exception as exc:

            print("[AI FRAME ERROR]", exc)

            cv2.putText(
                frame,
                "AI processing error",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

            annotated = frame

        # ----------------------------------------------------
        # SEND FRAME TO BROWSER
        # ----------------------------------------------------

        data = encode_frame(annotated)

        if data:
            yield data

        time.sleep(.03)

    cap.release()


# ============================================================
# ALERT SYSTEM
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

    # Keep only latest 8 alerts
    del alerts[8:]

    return item


# ============================================================
# MAIN PAGE
# ============================================================

@app.route("/")
def index():

    return render_template("index.html")


# ============================================================
# CAMERA API
# ============================================================

@app.get("/api/cameras")
def api_cameras():

    return jsonify(CAMERAS)


# ============================================================
# DASHBOARD API
# ============================================================

@app.get("/api/dashboard")
def api_dashboard():

    return jsonify({

        "system": {
            "feeds": len(CAMERAS),
            "latency": metrics["latency"],
            "mode": "prototype"
        },

        "metrics": metrics,

        "alerts": alerts,

        "timestamp": datetime.now().isoformat(
            timespec="seconds"
        )
    })


# ============================================================
# DEMO ALERT API
# ============================================================

@app.post("/api/demo-alert")
def api_demo_alert():

    camera = (
        request.json.get("sector", "Sector A-3")
        if request.is_json
        else "Sector A-3"
    )

    item = add_alert(
        random.choice(alert_types),
        camera,
        "AI event generated for demonstration",
        round(random.uniform(82, 98), 1)
    )

    return jsonify(item)


# ============================================================
# UPLOAD VIDEO API
# ============================================================

@app.post("/api/upload-video")
def upload_video():

    global latest_uploaded_video

    # --------------------------------------------------------
    # Check file
    # --------------------------------------------------------

    if "video" not in request.files:

        return jsonify({
            "error": "No video file was uploaded."
        }), 400

    file = request.files["video"]

    if file.filename == "":

        return jsonify({
            "error": "No video file selected."
        }), 400

    # --------------------------------------------------------
    # Allowed extensions
    # --------------------------------------------------------

    allowed_extensions = {
        "mp4",
        "mov",
        "avi",
        "mkv",
        "webm"
    }

    filename = secure_filename(file.filename)

    extension = filename.rsplit(".", 1)[-1].lower()

    if extension not in allowed_extensions:

        return jsonify({
            "error": "Unsupported video format."
        }), 400

    # --------------------------------------------------------
    # Save file
    # --------------------------------------------------------

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    filename = f"{timestamp}_{filename}"

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    try:

        file.save(filepath)

    except Exception as exc:

        print("[UPLOAD ERROR]", exc)

        return jsonify({
            "error": "Could not save uploaded video."
        }), 500

    # --------------------------------------------------------
    # Verify video
    # --------------------------------------------------------

    test_cap = cv2.VideoCapture(filepath)

    if not test_cap.isOpened():

        test_cap.release()

        try:
            os.remove(filepath)
        except Exception:
            pass

        return jsonify({
            "error": "The uploaded file is not a valid video."
        }), 400

    width = int(
        test_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        test_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    fps = test_cap.get(cv2.CAP_PROP_FPS)

    frame_count = int(
        test_cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    test_cap.release()

    # --------------------------------------------------------
    # Remember latest uploaded video
    # --------------------------------------------------------

    latest_uploaded_video = filepath

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

        "path": filepath,

        "width": width,

        "height": height,

        "fps": fps,

        "frames": frame_count,

        "message": "Video uploaded successfully."
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
    # If AI mode is requested and an uploaded video exists,
    # automatically use that uploaded video.
    # --------------------------------------------------------

    if mode == "ai":

        if latest_uploaded_video:

            if os.path.exists(latest_uploaded_video):

                source = latest_uploaded_video

                print(
                    "[VIDEO FEED] Using uploaded video:",
                    source
                )

            else:

                print(
                    "[VIDEO FEED] Uploaded video no longer exists."
                )

                source = "0"

        else:

            # No upload yet
            source = request.args.get(
                "source",
                "0"
            )

        generator = generate_ai_stream(
            source,
            camera_id
        )

    else:

        generator = generate_demo_stream(
            camera_id
        )

    return Response(
        generator,
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


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
