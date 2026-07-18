# STEP 1: Import Libraries
from IPython.display import display, Javascript
from google.colab.output import eval_js
from google.colab.patches import cv2_imshow
from base64 import b64decode
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import os

# STEP 2: Load Model and Labels
# Note: Adjust these paths if you place the model elsewhere in your Google Drive
MODEL_PATH = "/content/models/keras_model.h5"
LABEL_PATH = "/content/models/labels.txt"

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
class_names = open(LABEL_PATH, "r").readlines()
print("Model loaded successfully.")

# STEP 3: JavaScript Webcam Capture Function
def capture_image(filename='photo.jpg', quality=0.8):
  js_code = '''
    async function takePhoto(quality) {
      const div = document.createElement('div');
      const capture = document.createElement('button');
      capture.textContent = 'Capture';
      div.appendChild(capture);

      const video = document.createElement('video');
      video.style.display = 'block';
      const stream = await navigator.mediaDevices.getUserMedia({video: true});
      document.body.appendChild(div);
      div.appendChild(video);
      video.srcObject = stream;
      await video.play();

      google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

      await new Promise((resolve) => capture.onclick = resolve);
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      stream.getVideoTracks()[0].stop();
      div.remove();
      return canvas.toDataURL('image/jpeg', quality);
    }
    '''
  display(Javascript(js_code))
  data = eval_js('takePhoto({})'.format(quality))
  binary = b64decode(data.split(',')[1])
  
  with open(filename, 'wb') as f:
    f.write(binary)
  return filename

# STEP 4: Run Capture
try:
  filename = capture_image()
  print('Captured:', filename)
  
  # STEP 5: Preprocess and Predict
  # Teachable Machine requires 224x224 images
  img = Image.open(filename).resize((224, 224))
  img_array = np.asarray(img, dtype=np.float32)
  
  # Reshape into batch format (1, 224, 224, 3)
  img_array = img_array.reshape(1, 224, 224, 3)
  
  # Normalize the image
  img_array = (img_array / 127.5) - 1
  
  # Predict
  prediction = model.predict(img_array)
  class_id = np.argmax(prediction)
  sign = class_names[class_id].strip()
  confidence = float(np.max(prediction))
  
  # STEP 6: Show Output
  print(f"Predicted Sign: {sign}")
  print(f"Confidence: {confidence:.2f}")
  
  # STEP 7: Show Annotated Image
  img_cv = cv2.imread(filename)
  text = f"{sign} ({confidence:.2f})"
  cv2.putText(img_cv, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
  cv2_imshow(img_cv)
  
  # Clean up
  os.remove(filename)

except Exception as err:
  print(str(err))