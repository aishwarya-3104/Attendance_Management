import cv2
import os
import pandas as pd
from datetime import datetime
import face_recognition

# Directory paths
group_image_path = 'captured_image/Capture.PNG'
training_images_dir = 'Training_images'
individual_faces_dir = 'individual_faces'
attendance_csv_path = 'Attendance.csv'

# Load the group photo
group_image = cv2.imread(group_image_path)

# Convert the group image to RGB (face_recognition uses RGB images)
rgb_group_image = cv2.cvtColor(group_image, cv2.COLOR_BGR2RGB)

# Detect faces in the group image
face_locations = face_recognition.face_locations(rgb_group_image)

# Load training images and create encodings
known_face_encodings = []
known_face_names = []

for filename in os.listdir(training_images_dir):
    if filename.endswith('.png') or filename.endswith('.jpg'):
        image_path = os.path.join(training_images_dir, filename)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:  # Check if encodings are found
            known_face_encodings.append(encodings[0])
            known_face_names.append(os.path.splitext(filename)[0])

# Create a folder to save individual face images if it doesn't exist
if not os.path.exists(individual_faces_dir):
    os.makedirs(individual_faces_dir)

# Get current date and time
current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# Prepare the attendance data
attendance_data = []

# Loop through detected faces
for i, (top, right, bottom, left) in enumerate(face_locations):
    face_image = group_image[top:bottom, left:right]
    face_filename = f'{individual_faces_dir}/face_{i+1}.png'
    cv2.imwrite(face_filename, face_image)

    # Extract the face encoding for the current face
    face_encoding = face_recognition.face_encodings(rgb_group_image, [face_locations[i]])[0]

    # Compare face with known faces
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

    # Find the best match
    best_match_index = None
    if True in matches:
        best_match_index = face_distances.argmin()

    if best_match_index is not None and matches[best_match_index]:
        name = known_face_names[best_match_index]
    else:
        name = "Unknown"

    # Append the details to the attendance data
    attendance_data.append([name, current_datetime])

# Create or update the Attendance.csv file
attendance_df = pd.DataFrame(attendance_data, columns=['Name', 'DateTime'])

if os.path.exists(attendance_csv_path):
    # If the file exists, append the new data
    attendance_df.to_csv(attendance_csv_path, mode='a', header=False, index=False)
else:
    # If the file doesn't exist, create it with headers
    attendance_df.to_csv(attendance_csv_path, index=False)

print("Attendance has been recorded.")
