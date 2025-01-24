from ultralytics import YOLO
from process_video import process_video
from attack_AUP.applyUAP import apply_uap


if __name__ == "__main__":

    ogVideoPath = "./data/input/drive_short.mp4"
    attackedVideoPath = "./data/output/drive_fooled.mp4"
    visualize = True

    detector = YOLO("./model/traffic_sign_detector.pt", task="detect")
    detector.to('cuda')

    _, _, detections = process_video(ogVideoPath, detector, "initial", visualize)

    apply_uap(ogVideoPath, attackedVideoPath, detections)
    
    _, _, detections = process_video(attackedVideoPath, detector, "fooling", visualize)