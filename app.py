import pathlib
import cv2
import subprocess
import time
import os

# Path to the Haar cascade XML file for face detection
cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"

clf = cv2.CascadeClassifier(str(cascade_path))

camera = cv2.VideoCapture(0)

# Create a folder to store the captured images
folder_name = "captured_faces"
os.makedirs(folder_name, exist_ok=True)

# Set the time interval (in seconds) to capture images
capture_interval = 2
last_capture_time = 0

while True:
    _, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = clf.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    if len(faces) > 0 and time.time() - last_capture_time >= capture_interval:
        # Capture the image if there are faces and the interval has passed
        filename = f"image_{time.strftime('%Y%m%d%H%M%S')}.jpg"
        file_path = os.path.join(folder_name, filename)
        cv2.imwrite(file_path, frame)
        last_capture_time = time.time()
        print(f"Captured image: {file_path}")

    for (x, y, width, height) in faces:
        cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 0), 2)

    cv2.imshow("Faces", frame)

    if cv2.waitKey(1) == ord("q"):
        subprocess.run(["python", "app.py"])
        break

    # Reset the last capture time if no faces are detected
    if len(faces) == 0:
        last_capture_time = time.time()

camera.release()
cv2.destroyAllWindows()