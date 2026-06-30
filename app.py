import os
import base64
import time
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory
from predict import predict

app = Flask(__name__)

# Ensure we have a temp directory for uploads within the workspace
TEMP_DIR = Path("temp_uploads")
TEMP_DIR.mkdir(exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def api_predict():
    # Start timer for backend latency measurement
    start_time = time.perf_counter()
    
    image_path = None
    try:
        # Check if file was uploaded via multipart/form-data
        if "image" in request.files:
            file = request.files["image"]
            if file.filename == "":
                return jsonify({"error": "No selected file"}), 400
            
            # Save file to temp directory
            filename = f"upload_{int(time.time() * 1000)}.jpg"
            image_path = TEMP_DIR / filename
            file.save(image_path)
            
        # Check if image was sent as base64 in JSON payload (from camera)
        elif request.is_json and "image_b64" in request.json:
            data = request.json["image_b64"]
            if "," in data:
                # Remove header like 'data:image/jpeg;base64,'
                data = data.split(",")[1]
                
            image_data = base64.b64decode(data)
            filename = f"capture_{int(time.time() * 1000)}.jpg"
            image_path = TEMP_DIR / filename
            with open(image_path, "wb") as f:
                f.write(image_data)
        else:
            return jsonify({"error": "No image data provided"}), 400

        # Run prediction
        score = predict(str(image_path))
        
        # Calculate latency
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        
        # Determine classification
        # Score is probability of screen (1 = screen, 0 = real)
        is_screen = score >= 0.5
        label = "Screen Recapture" if is_screen else "Real Photo"
        confidence = score if is_screen else (1.0 - score)
        
        return jsonify({
            "success": True,
            "score": score,
            "is_screen": bool(is_screen),
            "label": label,
            "confidence": float(confidence),
            "latency_ms": float(latency_ms)
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
        
    finally:
        # Clean up temp file
        if image_path and os.path.exists(image_path):
            try:
                os.remove(image_path)
            except Exception:
                pass

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
