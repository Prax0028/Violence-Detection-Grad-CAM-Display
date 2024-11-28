from flask import Flask, request, render_template, send_file, url_for
import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from matplotlib import pyplot as plt

# Define the Flask app
app = Flask(__name__)

# Ensure upload and output folders exist
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Define the CNN model class
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )
        self.gradient = None

    def forward(self, images):
        x = self.feature_extractor(images)
        h = x.register_hook(self.activations_hook)
        x = self.maxpool(x)
        x = self.classifier(x)
        return x

    def activations_hook(self, grad):
        self.gradient = grad

    def get_activation_gradients(self):
        return self.gradient

    def get_activation(self, x):
        return self.feature_extractor(x)

# Load the model and weights
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CNNModel()
model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu'))['model_state_dict'])
model.to(DEVICE)
model.eval()

# Function to create Grad-CAM heatmap
def get_gradcam(model, image, size=224):
    model.eval()
    pred = model(image)
    pred_class = torch.argmax(pred, dim=1)
    model.zero_grad()
    pred[0, pred_class].backward()
    gradients = model.get_activation_gradients()
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    activations = model.get_activation(image).detach()

    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activations, dim=1).squeeze().cpu()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= torch.max(heatmap)
    heatmap = cv2.resize(heatmap.numpy(), (size, size))

    return heatmap

# Function to save frame and heatmap
def save_frame_and_heatmap(image, heatmap, predicted_class, frame_number):
    # Resize original image to match heatmap size
    image_resized = cv2.resize(image, (224, 224))

    # Add predicted class label to the original image
    classes = ['Non-Violent', 'Violent']
    label = f'Prediction: {classes[predicted_class]}'
    font = ImageFont.load_default()
    image_with_label = Image.fromarray(image_resized)
    draw = ImageDraw.Draw(image_with_label)
    draw.text((10, 10), label, fill='red', font=font)

    # Save the original image with label
    original_filename = f'original_frame_{frame_number}.jpg'
    original_path = os.path.join(app.config['OUTPUT_FOLDER'], original_filename)
    image_with_label.save(original_path)

    # Save the heatmap
    heatmap_filename = f'heatmap_frame_{frame_number}.jpg'
    heatmap_overlay_path = os.path.join(app.config['OUTPUT_FOLDER'], heatmap_filename)
    fig, ax = plt.subplots()
    ax.imshow(image_resized)
    ax.imshow(heatmap, cmap='jet', alpha=0.5)
    ax.axis('off')
    plt.savefig(heatmap_overlay_path, format='jpeg')
    plt.close(fig)

    return original_filename, heatmap_filename, label

# Route for uploading and processing the video
@app.route('/', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Process video
            cap = cv2.VideoCapture(filepath)
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            frame_count = 0
            output_images = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Process every nth frame for efficiency (adjust as needed)
                frame_count += 1
                if frame_count % 30 != 0:  # Adjust to control processing frequency
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                resized_frame = cv2.resize(frame_rgb, (224, 224))
                tensor_frame = transform(resized_frame).unsqueeze(0).to(DEVICE)

                # Predict using the model
                prediction = model(tensor_frame)
                predicted_class = torch.argmax(prediction, dim=1).item()

                # Generate Grad-CAM heatmap
                heatmap = get_gradcam(model, tensor_frame)

                # Save original frame and heatmap
                original_filename, heatmap_filename, label = save_frame_and_heatmap(
                    frame_rgb, heatmap, predicted_class, frame_count
                )
                output_images.append((original_filename, heatmap_filename, label))

            cap.release()
            return render_template('result.html', images=output_images)

    return render_template('upload.html')

# Route for serving output images
@app.route('/outputs/<filename>')
def serve_output_file(filename):
    return send_file(os.path.join(app.config['OUTPUT_FOLDER'], filename), mimetype='image/jpeg')

# Route for downloading results
@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(app.config['OUTPUT_FOLDER'], filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
