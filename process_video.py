# import libraries
import cv2

def process_video(video_path: str, detector, state: str, visualize: bool = True) -> tuple[int, float, tuple[int]]:
    """
    Processes a video to detect traffic signs using a given detector.
    Args:
        video_path (str): Path to the input video file.
        detector: The detection model to use for detecting traffic signs.
        state (str): The state of the detection process (e.g., 'initial', 'fooling').
        visualize (bool, optional): Whether to visualize the detection process. Defaults to True.
    Returns:
        tuple[int, float, list[tuple[int]]]: A tuple containing the total number of detections, 
                                             the average confidence of detections, 
                                             and a list of detected bounding boxes. (x1, y1, x2, y2, frameNum)
    """


    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Unable to open the input file: ", video_path)
        exit()
    ret = True

    confidences = []
    detectionsCount = 0
    detected_boxes: list[tuple[int]] = []
    frameCount = 0

    while ret:
        ret, frame = cap.read()
        
        detections = detector(frame, device='cuda')
        
        for detection in detections:
                for bbox in detection.boxes:

                    detectionsCount = detectionsCount + 1
                    confidences.append(bbox.conf[0]) # NOTE: might need casting to float

                    x1, y1, x2, y2 = bbox.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    detected_boxes.append((x1, y1, x2, y2, frameCount))

                    # Draw rectangle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        frameCount += 1
        try:
            cv2.imshow("Traffic sign detector", frame)
        except Exception:
                break
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(confidences) == 0:
        print("No detections were made")
        return 0, 0, []
    else:
        avgConfidence = sum(confidences) / len(confidences)
        avgConfidence = avgConfidence.item() * 100

    print("-----------------------------------")
    print(f"Stats {state} Fooling:")
    print(f"Total detections: ", detectionsCount)
    print(f"Average confidence: {round(avgConfidence, 1)}%")
    print("-----------------------------------")
    return detectionsCount, avgConfidence, detected_boxes


