import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time
from deep_sort_realtime.deepsort_tracker import DeepSort
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="AI Vehicle Counter",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load YOLOv4 Model (with caching)
@st.cache_resource
def load_yolo_model():
    yolo_net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
    layer_names = yolo_net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers()]
    
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    return yolo_net, output_layers, classes

# Load class names helper
def get_vehicle_type(class_id):
    return {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}.get(class_id, "other")

# Main processing function
def process_video(video_path, toll_line_y, buffer_zone, confidence_threshold):
    # Initialize
    yolo_net, output_layers, classes = load_yolo_model()
    tracker = DeepSort(max_age=50, n_init=5, max_iou_distance=0.7, max_cosine_distance=0.4)
    CLASS_IDS = [2, 3, 5, 7]  # car, motorcycle, bus, truck
    
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Create output writer
    output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file.name, fourcc, fps, (width, height))
    
    track_history = {}
    vehicle_counts = {"car": 0, "truck": 0, "bus": 0, "motorcycle": 0}
    frame_count = 0
    start_time = time.time()
    
    # Create Streamlit UI elements
    status_text = st.empty()
    video_placeholder = st.empty()
    progress_bar = st.progress(0)
    metrics_placeholder = st.empty()
    results_placeholder = st.empty()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        progress = min(frame_count / int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1.0)
        progress_bar.progress(progress)
        
        # Perform detection
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        yolo_net.setInput(blob)
        outputs = yolo_net.forward(output_layers)
        
        boxes, confs, class_ids = [], [], []
        for output in outputs:
            for det in output:
                scores = det[5:]
                class_id = np.argmax(scores)
                conf = scores[class_id]
                if class_id in CLASS_IDS and conf > confidence_threshold:
                    cx = int(det[0] * width)
                    cy = int(det[1] * height)
                    w = int(det[2] * width)
                    h = int(det[3] * height)
                    x, y = cx - w // 2, cy - h // 2
                    boxes.append([x, y, w, h])
                    confs.append(float(conf))
                    class_ids.append(class_id)
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(boxes, confs, confidence_threshold, 0.4)
        if len(indices) > 0:
            boxes = [boxes[i] for i in indices.flatten()]
            confs = [confs[i] for i in indices.flatten()]
            class_ids = [class_ids[i] for i in indices.flatten()]
        
        # Update tracker
        detections = list(zip(boxes, confs, class_ids))
        tracks = tracker.update_tracks(detections, frame=frame)
        
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            track_id = track.track_id
            ltrb = track.to_ltrb()
            class_id = track.get_det_class()
            vehicle = get_vehicle_type(class_id)
            if vehicle == "other":
                continue
                
            centroid_y = (ltrb[1] + ltrb[3]) // 2
            if track_id not in track_history:
                track_history[track_id] = {"counted": False, "prev_positions": []}
            track_history[track_id]["prev_positions"].append(centroid_y)
            if len(track_history[track_id]["prev_positions"]) > 5:
                track_history[track_id]["prev_positions"].pop(0)
                
            positions = track_history[track_id]["prev_positions"]
            direction = positions[-1] - positions[0] if len(positions) >= 2 else 0
            
            # Check if vehicle crossed the toll line
            if (centroid_y > (toll_line_y - buffer_zone) and 
                not track_history[track_id]["counted"] and 
                direction > 0 and centroid_y > toll_line_y + buffer_zone):
                vehicle_counts[vehicle] += 1
                track_history[track_id]["counted"] = True
                
            # Draw bounding box and ID
            color = (0, 255, 0) if track_history[track_id]["counted"] else (0, 0, 255)
            cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), color, 2)
            cv2.putText(frame, f"{vehicle} {track_id}", (int(ltrb[0]), int(ltrb[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw toll line and counts
        cv2.line(frame, (0, toll_line_y), (width, toll_line_y), (255, 0, 0), 2)
        cv2.putText(frame, "Toll Line", (10, toll_line_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Display counts on frame
        y_offset = 30
        for vehicle, count in vehicle_counts.items():
            cv2.putText(frame, f"{vehicle}: {count}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y_offset += 30
        
        # Write frame to output
        out.write(frame)
        
        # Display in Streamlit (convert BGR to RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", caption="Real-time Vehicle Tracking")
        
        # Update status and metrics
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        status_text.info(f"Processing: Frame {frame_count} | {fps:.1f} FPS")
        
        # Display metrics
        metrics_placeholder.subheader("Live Counts")
        cols = metrics_placeholder.columns(4)
        for i, (vehicle, count) in enumerate(vehicle_counts.items()):
            cols[i].metric(vehicle.capitalize(), count)
    
    # Release resources
    cap.release()
    out.release()
    
    # Final results
    processing_time = time.time() - start_time
    results_placeholder.success("✅ Processing complete!")
    results_placeholder.subheader("Final Results")
    results_placeholder.dataframe(
        pd.DataFrame(list(vehicle_counts.items()), columns=["Vehicle Type", "Count"]),
        width=400
    )
    results_placeholder.write(f"Total processing time: {processing_time:.2f} seconds")
    
    return output_file.name, vehicle_counts

# Main Streamlit app
def main():
    st.title("🚗 AI-Powered Vehicle Counter")
    st.markdown("""
    **Track and count vehicles in real-time** using YOLOv4 for detection and DeepSORT for tracking.
    Adjust parameters in the sidebar for optimal performance.
    """)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        uploaded_file = st.file_uploader("Upload Traffic Video", type=["mp4", "avi", "mov"])
        toll_line_y = st.slider("Toll Line Position", 0, 1000, 500, 10,
                               help="Vertical position of the counting line")
        buffer_zone = st.slider("Buffer Zone", 5, 100, 25, 5,
                               help="Zone around toll line to prevent double-counting")
        confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05,
                                        help="Minimum confidence for vehicle detection")
        
        st.header("ℹ️ About")
        st.markdown("""
        This system uses:
        - **YOLOv4** for vehicle detection
        - **DeepSORT** for object tracking
        - **Streamlit** for real-time visualization
        
        Detected vehicle types: Cars, Trucks, Buses, Motorcycles
        """)
    
    # Main content
    if uploaded_file is not None:
        # Save uploaded file to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        
        st.subheader("Processing Video...")
        output_file, counts = process_video(
            video_path, 
            toll_line_y, 
            buffer_zone, 
            confidence_threshold
        )
        
        # Show download link for processed video
        st.subheader("Processed Video")
        with open(output_file, "rb") as f:
            st.download_button(
                label="Download Processed Video",
                data=f,
                file_name="processed_video.mp4",
                mime="video/mp4"
            )
        
        # Clean up temporary files
        os.unlink(video_path)
        os.unlink(output_file)
    else:
        # Show demo video before upload
        st.subheader("Video")
        st.video("Example.mp4")  # Replace with actual demo URL
        st.info("Please upload a video file to begin processing")

if __name__ == "__main__":
    main()