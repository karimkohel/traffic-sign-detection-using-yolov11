# import libraries
from ultralytics import YOLO
import cv2

detector = YOLO("./model/traffic_sign_detector.pt", task="detect")

def run_video(video_path: str, state: str, visualize: bool = True) -> tuple[int, float]:


    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Unable to open the input file")
        exit()
    ret = True

    confidences = []
    detectionsCount = 0

    while ret:
        ret, frame = cap.read()
        
        detections = detector(frame)
        
        for detection in detections:
                for bbox in detection.boxes:

                    detectionsCount = detectionsCount + 1
                    confidences.append(bbox.conf[0]) # NOTE: might need casting to float

                    x1, y1, x2, y2 = bbox.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Draw rectangle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
                cv2.imshow("Traffic sign detector", frame)
        
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cap.release()
    cv2.destroyAllWindows()
    avgConfidence = sum(confidences) / len(confidences)
    print("-----------------------------------")
    print(f"Stats {state} Fooling:")
    print(f"Total detections: ", detectionsCount)
    print(f"Average confidence: ", avgConfidence)
    print("-----------------------------------")
    return detectionsCount, avgConfidence


if __name__ == "__main__":

    video_path = "./data/input/traffic_signs.mp4"
    visualize = True
    
    run_video(video_path, "Before", visualize)
    run_video(video_path, "After", visualize)