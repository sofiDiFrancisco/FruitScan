import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Define the class names (must match the order used during training)
class_names = ['freshapples', 'freshbanana', 'freshoranges', 'rottenapples', 'rottenbanana', 'rottenoranges']

def load_model(model_path, num_classes):
    """Loads the pre-trained ResNet model."""
    model = models.resnet34(pretrained=False) # Load with pretrained=False as we load state_dict
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))) # Load to CPU
        model.eval() # Set model to evaluation mode
        print(f"Model loaded successfully from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        raise # Re-raise the exception to be caught by the calling function

    return model

def preprocess_image(image: Image.Image):
    """Preprocesses the input image for the model."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0) # Add batch dimension

def predict_image(model, image_tensor):
    """Makes a prediction using the loaded model."""
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted_class_index = torch.max(outputs, 1)

    predicted_class_name = class_names[predicted_class_index.item()]
    return predicted_class_name

def get_fruit_info(predicted_class_name):
    """Simulates fetching additional information about the fruit."""
    fruit_info_map = {
        'freshapples': "This apple appears fresh and ready to eat!",
        'freshbanana': "This banana is fresh and looks delicious!",
        'freshoranges': "This orange is fresh and juicy!",
        'rottenapples': "This apple appears rotten and should not be consumed.",
        'rottenbanana': "This banana is rotten and not suitable for eating.",
        'rottenoranges': "This orange is rotten and should be discarded."
    }
    return fruit_info_map.get(predicted_class_name, "Could not retrieve additional information for this fruit.")


if __name__ == '__main__':
    # Example usage (optional, for testing utilities)
    # Assuming you have a test image named 'test_image.jpg' in the same directory
    try:
        test_image_path = 'test_image.jpg' # Replace with a real image path if testing
        dummy_model_path = 'modelo_resnet34.pth' # Replace with your model path

        # Create a dummy model for testing purposes if the model file doesn't exist yet
        if not os.path.exists(dummy_model_path):
            print(f"Creating a dummy model file for testing utilities: {dummy_model_path}")
            dummy_model = models.resnet34(pretrained=False)
            dummy_model.fc = nn.Linear(dummy_model.fc.in_features, len(class_names))
            torch.save(dummy_model.state_dict(), dummy_model_path)

        # Load a dummy image or create one for testing
        try:
            test_image = Image.open(test_image_path).convert('RGB')
        except FileNotFoundError:
             print(f"Test image not found at {test_image_path}. Creating a dummy image for testing.")
             # Create a dummy black image
             test_image = Image.new('RGB', (224, 224), color = 'black')


        loaded_model = load_model(dummy_model_path, len(class_names))
        processed_image = preprocess_image(test_image)
        prediction = predict_image(loaded_model, processed_image)
        print(f"Test prediction: {prediction}")

        # Test the new function
        fruit_info = get_fruit_info(prediction)
        print(f"Fruit Info: {fruit_info}")

    except FileNotFoundError:
        print("Dummy model file not found. Cannot run utility test.")
    except Exception as e:
        print(f"An error occurred during utility testing: {e}")
