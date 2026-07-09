import cv2
import time
from ultralytics import YOLO

def main():
    print("Loading YOLO...")
    model = YOLO("yolo11n.pt")
    
    print("Opening camera 0...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open camera.")
        return
        
    cv2.namedWindow("Test", cv2.WINDOW_NORMAL)
    
    prev_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            start_yolo = time.time()
            results = model(frame, verbose=False)
            end_yolo = time.time()
            
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time + 1e-6)
            prev_time = curr_time
            
            yolo_ms = (end_yolo - start_yolo) * 1000
            
            cv2.putText(frame, f"FPS: {fps:.1f} | YOLO: {yolo_ms:.1f}ms", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                       
            cv2.imshow("Test", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
