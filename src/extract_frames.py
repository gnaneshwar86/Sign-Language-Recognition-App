import cv2
import os

VIDEO_DIR = "dataset/raw_videos"
FRAME_DIR = "dataset/frames"
FRAME_RATE = 5   # take 1 frame every 5 frames

os.makedirs(FRAME_DIR, exist_ok=True)

for label in os.listdir(VIDEO_DIR):
    label_path = os.path.join(VIDEO_DIR, label)
    frame_label_path = os.path.join(FRAME_DIR, label)
    os.makedirs(frame_label_path, exist_ok=True)

    for video in os.listdir(label_path):
        
        video_path = os.path.join(label_path, video)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("Skipping broken video:", video_path)
            continue


        count = 0
        saved = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if count % FRAME_RATE == 0:
                frame = cv2.resize(frame, (224, 224))
                cv2.imwrite(
                    f"{frame_label_path}/{video}_{saved}.jpg",
                    frame
                )
                saved += 1
            count += 1

        cap.release()

print("✅ Frame extraction completed")
