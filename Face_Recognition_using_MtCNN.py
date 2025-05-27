# !pip install facenet-pytorch
# !pip install opencv-python
import os
import cv2
import time
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image
import skfuzzy as fuzz
import numpy as np

# Initialize MTCNN and InceptionResnetV1
mtcnn0 = MTCNN(image_size=240, margin=0, keep_all=False, min_face_size=40)
mtcnn = MTCNN(image_size=240, margin=0, keep_all=True, min_face_size=40)
resnet = InceptionResnetV1(pretrained='vggface2').eval() 

# Initialize empty lists to store embeddings and names
embedding_list = []
name_list = []

# Process images in the photos folder
photos_folder = "photos"  # Path to the folder containing images
for filename in os.listdir(photos_folder):
    image_path = os.path.join(photos_folder, filename)
    if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Process each image for embeddings
        img = Image.open(image_path)
        face, prob = mtcnn0(img, return_prob=True)

        if face is not None and prob > 0.92:
            emb = resnet(face.unsqueeze(0)) # Dimension
            embedding_list.append(emb.detach()) # Tensor
            name_list.append(filename.split('.')[0])  # Use the filename (without extension) as the name

# Save the data (embeddings and names)
data = [embedding_list, name_list]
torch.save(data, 'data.pt')  # Save data.pt file

# Fuzzy Logic Membership Functions for Distance and Confidence
def fuzzy_distance(distance):
    dist_range = np.linspace(0, 2, 100)

    # Distance fuzzy sets (Close, Medium, Far)
    close = fuzz.trimf(dist_range, [0, 0, 0.5])
    medium = fuzz.trimf(dist_range, [0, 0.5, 1.0])
    far = fuzz.trimf(dist_range, [0.5, 1.0, 2.0])
    
    # Calculate the membership values for the given distance
    close_membership = fuzz.interp_membership(dist_range, close, distance)
    medium_membership = fuzz.interp_membership(dist_range, medium, distance)
    far_membership = fuzz.interp_membership(dist_range, far, distance)
    
    return close_membership, medium_membership, far_membership

def fuzzy_confidence(confidence):
    conf_range = np.linspace(0, 1, 100)
    
    # Confidence fuzzy sets (Low, Medium, High)
    low = fuzz.trimf(conf_range, [0, 0, 0.5])
    medium = fuzz.trimf(conf_range, [0, 0.5, 1.0])
    high = fuzz.trimf(conf_range, [0.5, 1.0, 1.0])
    
    # Calculate the membership values for the given confidence
    low_membership = fuzz.interp_membership(conf_range, low, confidence)
    medium_membership = fuzz.interp_membership(conf_range, medium, confidence)
    high_membership = fuzz.interp_membership(conf_range, high, confidence)
    
    return low_membership, medium_membership, high_membership

# Load the saved data
load_data = torch.load('data.pt')
embedding_list = load_data[0]
name_list = load_data[1]

# Open the webcam
cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame, try again")
        break

    # Convert frame to image for MTCNN processing
    img = Image.fromarray(frame)
    img_cropped_list = None  # Initialize to default
    match_probability = 0.0  # Initialize to default

    try:
        img_cropped_list, prob_list = mtcnn(img, return_prob=True)
    except Exception as e:
        print(f"Error during face detection: {e}")

    if img_cropped_list is not None:
        boxes, _ = mtcnn.detect(img)

        for i, prob in enumerate(prob_list):
            if prob > 0.90:
                emb = resnet(img_cropped_list[i].unsqueeze(0)).detach()

                # List of distances for face recognition
                dist_list = [torch.dist(emb, emb_db).item() for emb_db in embedding_list]

                min_dist = min(dist_list)  # Get minimum distance
                min_dist_idx = dist_list.index(min_dist)  # Get index of minimum distance
                name = name_list[min_dist_idx]  # Get name corresponding to the minimum distance

                # Get bounding box for detected face
                box = boxes[i]
                original_frame = frame.copy()  # Store the frame before drawing on it

                # Apply fuzzy logic to determine the matching probability
                close_membership, medium_membership, far_membership = fuzzy_distance(min_dist)
                low_membership, medium_conf_membership, high_membership = fuzzy_confidence(prob)

                # Compute weighted average of memberships
                match_probability = (close_membership * high_membership) + \
                                    (medium_membership * medium_conf_membership) + \
                                    (far_membership * low_membership)

                # Scale the match probability to be in the range of [0, 1]
                match_probability = np.clip(match_probability, 0, 1)

                # Draw the name and match probability on the frame
                frame = cv2.putText(frame, f'{name} Match Probability: {match_probability:.2f}', 
                                    (int(box[0]), int(box[1])), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # Draw the bounding box around the face
                frame = cv2.rectangle(frame, (int(box[0]), int(box[1])), 
                                      (int(box[2]), int(box[3])), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Output", frame)

    k = cv2.waitKey(1)
    if k % 256 == 27:  # ESC key to exit
        break

    # Save the image of the recognized person if match probability is high
    if match_probability > 0.8 and 'name' in locals() and not os.path.exists(f'photos/{name}'):
        os.mkdir(f'photos/{name}')

        img_name = f"photos/{name}/{int(time.time())}.jpeg"
        cv2.imwrite(img_name, original_frame)
        print(f"Saved: {img_name}")

# Release the webcam and close windows
cam.release()
cv2.destroyAllWindows()
