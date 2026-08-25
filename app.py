from flask import Flask, render_template, jsonify, Response, request
import cv2
import time
import random
import threading
from datetime import datetime
import os

app = Flask(__name__)

# The original AI modules are kept intact and are loaded only when AI mode is requested.
ai_engine = None
ai_lock = threading.Lock()

CAMERAS = [
    {"id": "A3", "name": "Sector A-3", "type": "Thermal Camera", "status": "ONLINE"},
    {"id": "B1", "name": "Sector B-1", "type": "Fence Perimeter", "status": "ONLINE"},
    {"id": "C4", "name": "Sector C-4", "type": "Checkpoint", "status": "ONLINE"},
]

alert_types = [
    "Perimeter Intrusion", "Thermal Anomaly", "Vehicle Trespassing",
    "Unidentified Target", "Suspicious Activity"
]
sectors = ["Sector A-3", "Sector B-1", "Sector C-4"]
alerts = [
    {"type": "Perimeter Intrusion", "sector": "Sector A-3", "details": "Target spotted near the fence", "confidence": 94.2, "time": "10:42:15 AM"}
]
metrics = {"fps": 30, "humans": 1, "vehicles": 0, "tracks": 1, "latency": 14, "bandwidth": 85}


def now_time():
    return datetime.now().strftime("%I:%M:%S %p")


def make_demo_frame(camera_id="A3"):
    """Generate a deterministic-looking demo CCTV frame so the website works without a camera."""
    w, h = 960, 540
    frame = __import__('numpy').zeros((h, w, 3), dtype='uint8')
    frame[:] = (8, 12, 22)

    # grid
    for x in range(0, w, 40):
        cv2.line(frame, (x, 0), (x, h), (22, 31, 46), 1)
    for y in range(0, h, 40):
        cv2.line(frame, (0, y), (w, y), (22, 31, 46), 1)

    cam = next((c for c in CAMERAS if c['id'] == camera_id), CAMERAS[0])
    cv2.putText(frame, f"LIVE  |  {cam['name']} ({cam['type']})", (24, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (230, 235, 245), 2)
    cv2.putText(frame, "IBVAP AI ANALYTICS  •  PROTOTYPE MODE", (24, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (70, 220, 170), 1)

    # demo target / bounding box
    bx, by, bw, bh = 390, 150, 150, 230
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (40, 90, 245), 2)
    cv2.rectangle(frame, (bx, by - 25), (bx + 165, by), (40, 90, 245), -1)
    cv2.putText(frame, "PERSON  94.2%  ID:17", (bx + 6, by - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1)

    # virtual fence
    cv2.line(frame, (80, 410), (880, 410), (50, 70, 245), 2)
    cv2.putText(frame, "VIRTUAL FENCE", (410, 398), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 100, 250), 1)

    cv2.rectangle(frame, (20, h - 48), (w - 20, h - 18), (4, 8, 16), -1)
    cv2.putText(frame, f"HUMANS: {metrics['humans']}   VEHICLES: {metrics['vehicles']}   TRACKS: {metrics['tracks']}   FPS: {metrics['fps']}",
                (30, h - 27), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 220, 235), 1)
    return frame


def load_ai():
    global ai_engine
    with ai_lock:
        if ai_engine is not None:
            return ai_engine
        try:
            from detection.human_vehicle_detector import HumanVehicleDetector
            from anpr.anpr_pipeline import ANPRPipeline
            from suspicious.pose_detector import SuspiciousPoseDetector
            ai_engine = {
                "detector": HumanVehicleDetector(),
                "anpr": None,
                "suspicious": SuspiciousPoseDetector(),
            }
            ai_engine["anpr"] = ANPRPipeline(ai_engine["detector"])
            return ai_engine
        except Exception as exc:
            print("[AI MODE ERROR]", exc)
            return None


def generate_demo_stream(camera_id):
    while True:
        frame = make_demo_frame(camera_id)
        ok, jpg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if ok:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n')
        time.sleep(1 / 15)


def generate_ai_stream(source, camera_id):
    engine = load_ai()
    if engine is None:
        yield from generate_demo_stream(camera_id)
        return

    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    if not cap.isOpened():
        yield from generate_demo_stream(camera_id)
        return

    frame_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
            if not ok:
                break

        frame_count += 1
        try:
            tracked = engine["detector"].process_frame(frame)
            engine["anpr"].process_frame(frame, tracked)
            pose_alert = engine["suspicious"].process_frame(frame)
            annotated = engine["detector"].annotate_frame(frame, tracked)

            humans = sum(o.category == 'human' for o in tracked)
            vehicles = sum(o.category == 'vehicle' for o in tracked)
            metrics.update({"humans": humans, "vehicles": vehicles, "tracks": len(tracked), "fps": int(cap.get(cv2.CAP_PROP_FPS) or 25)})

            if pose_alert:
                add_alert("Suspicious Activity", next(c['name'] for c in CAMERAS if c['id'] == camera_id), "Pose anomaly detected", 88.0)
                cv2.rectangle(annotated, (10, 10), (420, 55), (0, 0, 255), -1)
                cv2.putText(annotated, "ALERT: SUSPICIOUS ACTIVITY", (20, 42), cv2.FONT_HERSHEY_SIMPLEX, .65, (255,255,255), 2)
        except Exception as exc:
            cv2.putText(frame, f"AI processing error: {str(exc)[:70]}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 100, 255), 2)
            annotated = frame

        ok, jpg = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if ok:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n')
        time.sleep(.03)

    cap.release()


def add_alert(alert_type, sector, details, confidence=None):
    item = {"type": alert_type, "sector": sector, "details": details,
            "confidence": confidence, "time": now_time()}
    alerts.insert(0, item)
    del alerts[8:]
    return item


@app.route('/')
def index():
    return render_template('index.html')


@app.get('/api/cameras')
def api_cameras():
    return jsonify(CAMERAS)


@app.get('/api/dashboard')
def api_dashboard():
    return jsonify({
        "system": {"feeds": len(CAMERAS), "latency": metrics["latency"], "mode": "prototype"},
        "metrics": metrics,
        "alerts": alerts,
        "timestamp": datetime.now().isoformat(timespec='seconds')
    })


@app.post('/api/demo-alert')
def api_demo_alert():
    camera = request.json.get('sector', 'Sector A-3') if request.is_json else 'Sector A-3'
    item = add_alert(random.choice(alert_types), camera, "AI event generated for demonstration", round(random.uniform(82, 98), 1))
    return jsonify(item)


@app.get('/video_feed')
def video_feed():
    camera_id = request.args.get('camera', 'A3')
    mode = request.args.get('mode', 'demo')
    source = request.args.get('source', '0')
    generator = generate_ai_stream(source, camera_id) if mode == 'ai' else generate_demo_stream(camera_id)
    return Response(generator, mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
