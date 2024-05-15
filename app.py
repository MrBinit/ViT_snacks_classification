import torch
from torchvision import transforms
from flask import Flask, request, render_template, jsonify
from PIL import Image

# Error handling for invalid image uploads
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app = Flask(__name__)

# Load the trained model (assuming it's a dictionary or has a `load` method)
def load_model():
    model = torch.load("trained_model.pt", map_location=torch.device('cpu'))
    # If your model has a specific loading method:
    # model.load_state_dict(...)  # Replace with your model's loading method
    return model

# Define prediction function (modify as needed based on your model)
def predict(img, model):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
        ])

    img = transform(img)
    # Assuming your model has a 'forward' pass function:
    output = model(img.unsqueeze(0).to(device))
    prediction = output.argmax(dim=1).item()

    # Assuming labels (id to class name mapping) are available
    predicted_class = id2label[prediction]  # Modify as needed
    return predicted_class

# Load the model outside the route function for efficiency
loaded_model = load_model()

@app.route("/", methods=["GET", "POST"])

@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        # Check if a file is uploaded
        if 'image' not in request.files:
            return render_template("upload.html", message="No file selected")

        # Access the uploaded file using request.files['image']
        file = request.files['image']

        # Check if the file is an image
        if file.filename == '':
            return render_template("upload.html", message="No file selected")
        if file and allowed_file(file.filename):
            try:
                # Process image
                image = Image.open(file).convert('RGB')

                # Call the prediction function (assuming it's correct for your model)
                predicted_class = predict(image, loaded_model)
                return render_template("predict.html", prediction=predicted_class, image_path=file.filename)
            except Exception as e:
                return render_template("upload.html", message=f"Error: {str(e)}")
        else:
            return render_template("upload.html", message="Only PNG, JPG or JPEG files allowed")

    else:
        return render_template("upload.html")

if __name__ == "__main__":
    app.run(debug=True)