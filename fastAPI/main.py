from fastapi import FastAPI, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketDisconnect
from PIL import Image
import torch
import io
import os
import cv2
import numpy as np
import shutil
import json
import base64
import asyncio

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*autocast.*")

TEMP_VIDEO_DIR = "./temp_video"  # Temporary directory for video files
CONFIDENCE_THRESHOLD = 0.5  # Confidence threshold for detections
IOU_THRESHOLD = 0.45  # IoU threshold for non-maximum suppression
HIGH_SEVERITY_THRESHOLD = 2  # Number of detections to consider severity "high"
os.makedirs(TEMP_VIDEO_DIR, exist_ok=True)


# Directory for YOLO output
YOLO_OUTPUT_DIR = "./runs/detect"
os.makedirs(YOLO_OUTPUT_DIR, exist_ok=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files to serve YOLO detection images
app.mount("/runs", StaticFiles(directory="runs"), name="runs")

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='./model/best.pt', device='cpu')

@app.get("/")
async def root():
    return {"message": "Welcome to the Pothole Detection API"}

###############################################################################

@app.post("/analyze-pothole")
async def analyze_pothole(file: UploadFile):
    # Validate file type for images
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a JPEG or PNG image.")

    # Load image from uploaded file
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))

    # Perform pothole detection using YOLOv5
    results = model(img)
    results.save(YOLO_OUTPUT_DIR)

    # Get the latest YOLO output folder
    exp_folders = sorted(
        [d for d in os.listdir(YOLO_OUTPUT_DIR) if os.path.isdir(os.path.join(YOLO_OUTPUT_DIR, d))],
        key=lambda x: os.path.getmtime(os.path.join(YOLO_OUTPUT_DIR, x))
    )
    latest_exp_path = os.path.join(YOLO_OUTPUT_DIR, exp_folders[-1])

    # Assume the labeled image is named "image0.jpg"
    labeled_image_name = "image0.jpg"
    labeled_image_path = os.path.join(latest_exp_path, labeled_image_name)

    if not os.path.exists(labeled_image_path):
        raise HTTPException(status_code=404, detail="Labeled image not found.")

    # Analyze detection results
    num_detections = results.xyxy[0].shape[0]
    severity = "low" if num_detections < 3 else "high"

    # Clean up the 'runs/detect' folder if it has more than 50 subdirectories
    exp_folders = sorted(
        [d for d in os.listdir(YOLO_OUTPUT_DIR) if os.path.isdir(os.path.join(YOLO_OUTPUT_DIR, d))],
        key=lambda x: os.path.getmtime(os.path.join(YOLO_OUTPUT_DIR, x))
    )

    if len(exp_folders) >= 50:
        # Delete all subdirectories in 'runs/detect/'
        for folder in exp_folders:
            folder_path = os.path.join(YOLO_OUTPUT_DIR, folder)
            shutil.rmtree(folder_path)
            print(f"Deleted {folder_path} due to reaching the maximum number of folders.")

    return {
        "severity": severity,
        "num_potholes_detected": num_detections,
        "labeled_image_url": f"/runs/detect/{exp_folders[-1]}/{labeled_image_name}"
    }


#######################################################


class CentroidTracker:  # (Same CentroidTracker class as before)
    def __init__(self, maxDisappeared=50):
        self.nextObjectID = 0
        self.objects = {}
        self.disappeared = {}
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1
        return self.nextObjectID - 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def euclidean_distance(self, point1, point2):
        """Calculates the Euclidean distance between two points."""
        return np.sqrt(np.sum((np.array(point1) - np.array(point2))**2))

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # Manual distance calculation (replace cdist)
            D = np.zeros((len(objectCentroids), len(inputCentroids)))
            for i in range(len(objectCentroids)):
                for j in range(len(inputCentroids)):
                    D[i, j] = self.euclidean_distance(objectCentroids[i], inputCentroids[j])

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        return self.objects

# Initialize tracker
tracker = CentroidTracker()


def assess_severity(frame, bbox):
    """
    Placeholder for severity assessment.  Replace with your logic.
    """
    x1, y1, x2, y2 = map(int, bbox)  # Ensure integers for indexing
    pothole_roi = frame[y1:y2, x1:x2]

    if pothole_roi.size == 0:  # Handle empty ROI if bbox is out of bounds
        return "Unknown"  # Or some default value

    # Example:  Simple area-based assessment (replace with more sophisticated methods)
    area = (x2 - x1) * (y2 - y1)
    if area < 5000:
        return "Low"
    elif area < 15000:
        return "Medium"
    else:
        return "High"

def process_frame(frame):
    """
    Processes a single frame, performs pothole detection, and returns the
    base64 encoded image and metadata.
    """
    try:
        # Pothole Detection (YOLO)
        if model is None:
            print("Model is None, skipping detection")
            _, buffer = cv2.imencode(".jpg", frame)
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            return jpg_as_text, {"pothole_count": 0, "average_severity": 0}  # Send a default image if detection fails


        results = model(frame)

        # Extract bounding boxes
        bboxes = []
        confidences = []

        if hasattr(results, 'xyxy') and len(results.xyxy) > 0: #Check that we can access this attribute
          for *xyxy, conf, cls in results.xyxy[0]: # Get results
              x1, y1, x2, y2 = map(int, xyxy) # Extract the coordinates
              bboxes.append((x1, y1, x2, y2))
              confidences.append(float(conf))
        else:
          print("No objects detected")
          bboxes = []
          confidences = []


        # Update tracker
        objects = tracker.update(bboxes)

        # Prepare data for sending
        pothole_data = []
        pothole_count = 0  # Initialize the counter
        total_severity = 0  # New: Accumulate severity values

        for (objectID, centroid) in objects.items():
          pothole_count += 1
          #Find the bounding box associated with this centroid
          x1, y1, x2, y2 = -1, -1, -1, -1
          for i, box in enumerate(bboxes):
            cx = int((box[0] + box[2]) / 2.0)
            cy = int((box[1] + box[3]) / 2.0) # Corrected line
            if abs(centroid[0] - cx) < 5 and abs(centroid[1] - cy) < 5:
                x1, y1, x2, y2 = box
                break
          if x1 != -1:

            severity = assess_severity(frame, (x1, y1, x2, y2))
            print(f"Severity: {severity}")

            # Convert severity to a numerical value for averaging
            severity_value = 0
            if severity == "Low":
                severity_value = 1
            elif severity == "Medium":
                severity_value = 2
            elif severity == "High":
                severity_value = 3

            total_severity += severity_value


        # Create metadata
        average_severity = 0
        if pothole_count > 0:
          average_severity = total_severity / pothole_count

        metadata = {
            "pothole_count": pothole_count,
            "average_severity": average_severity
        }


        # Draw bounding boxes on the frame (example)
        for (x1, y1, x2, y2) in bboxes: #Draw bboxes
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Encode the processed frame as a JPEG image
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')

        return jpg_as_text, metadata  # Return both image and metadata

    except Exception as e:
        print(f"Error processing frame: {e}")
        _, buffer = cv2.imencode(".jpg", frame)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        return jpg_as_text, {"pothole_count": 0, "average_severity": 0} # Return a default image if processing fails

@app.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()
            image = Image.open(io.BytesIO(data))
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # Process the frame and get both the image and metadata
            processed_image_b64, metadata = process_frame(frame)

            # Create the JSON payload with both image and metadata
            payload = {"image": processed_image_b64, **metadata}

            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(0.001) #VERY IMPORTANT - Avoid Overloading the websocket

    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        print("WebSocket closed.")