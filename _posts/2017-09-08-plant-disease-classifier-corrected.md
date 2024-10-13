---
layout: post
title: "How to Build a Plant Disease Classifier App That Knows More About Your Plants Than You Do"
date: 2017-09-08
categories: blog
---

## Introduction

Ever stared at your plants and thought, *"What’s wrong with you this time?"*—only to wish they could just tell you? Well, today, we’re building an app that does (kinda) just that! In this tutorial, we’ll create a plant disease classifier that can identify whether your plant is rocking that healthy glow or is struggling with something that sounds like it came straight out of a sci-fi movie.

We’ll use **PyTorch** (a fancy deep learning library) to train the model and **Streamlit** to make a shiny web app so you can show off to your friends how tech-savvy you are. Let’s get our hands dirty—digitally, of course.

---

### Tech We’re Using:

- **PyTorch**: Because building neural networks from scratch is way too much work.
- **Streamlit**: Makes it look like we actually know how to code a web app. (Spoiler: It’s super easy!)
- **PIL**: Not a fancy drink, just a library for handling images.
- **Torchvision**: For transforming images so they look the way our model wants them to.

---

### What’s the Plan?

Here’s the super official breakdown:

1. **Build the Brain**: We’ll design a neural network that can recognize plant diseases.
2. **Preprocess the Image**: Clean up the plant selfie before passing it to the model.
3. **Load and Use the Model**: Dust off the pre-trained model and put it to work.
4. **Create the Streamlit App**: Wrap everything up in a neat interface that even a plant newbie can use.

---

## 1. Designing the Brain (Neural Network Architecture)

Let’s start with the brains of the operation—a Convolutional Neural Network (CNN). This isn’t your average garden variety network; it’s trained to recognize plant problems like it’s been gardening its whole life.

```python
class NeuralNet(nn.Module):
    def __init__(self, n_labels):
        super(NeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        # More layers of magic stuff follow...

```

Translation:

    Convolutional Layers: Think of these as layers that are really good at spotting shapes, like diseased leaf patterns.
    Batch Normalization: Makes sure things don’t get too wild during training. It’s like a chill pill for our layers.
    Pooling Layers: These shrink down the image size, making sure the network doesn't work overtime.
    Fully Connected Layers: These guys are the final judges, deciding if your plant is A-OK or having a crisis.
    Dropout: Drops random neurons like it’s hot—keeps the model from getting too attached to any one pattern.

2. Prepping the Leaf Selfie

Plants don’t know how to take perfect selfies, so we have to help out. We’ll resize the image, turn it grayscale (because plants are colorblind anyway), and transform it into a format our neural network actually understands.

```python

def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    input_tensor = preprocess(image)
    input_tensor = input_tensor.unsqueeze(0)
    return input_tensor
```
Why?

Because the model is picky. It wants grayscale images in a specific size and format. Think of it like asking for a perfect cup of coffee—no foam, extra hot.
3. Loading the Plant Whisperer Model

Our neural network can’t read your mind (yet), so we need to load a pre-trained model file. And if you’re lucky enough to have a GPU, your model will run much faster. If not, no worries—CPU works too (just don’t expect lightning speed).

```python

def load_model(path, num_classes):
    model = NeuralNet(num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()  
    return model
```
4. Bringing It All Together in Streamlit

Finally, we get to the fun part—making the app. This is where Streamlit shines. We’re gonna build an interface that lets you upload an image and click a button like a pro.

```python

def main():
    st.title('Plant Disease Classifier - Is Your Plant in Trouble?')
    st.text('Upload a leaf image, and we’ll tell you if your plant is thriving or needs some TLC.')

    label_dict = {
        0: "Apple scab", 1: "Black rot", 2: "Cedar apple rust",
        3: "Healthy", 4: "Powdery mildew", 5: "Spot",
        6: "Corn (Maize) Common Rust", 7: "Blight", 8: "Grape Esca (Black Measles)",
        9: "Orange Haunglongbing (Citrus greening)", 10: "Strawberry Leaf scorch",
        11: "Tomato Leaf Mold", 12: "Tomato Spider mites", 13: "Tomato Yellow Leaf Curl Virus",
        14: "Tomato mosaic virus"
    }

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
        except IOError:
            st.error("Uh-oh, that file doesn’t look right. Try again with an image.")
            return
        
        model = load_model('best_model-1.pth', 15)
        
        if st.button('Predict'):
            class_name, probability = predict_image_class(image, model, label_dict)
            st.write(f'**Diagnosis**: {class_name} with confidence: {probability:.2f}')
```

What’s Happening Here?

    File Uploader: This little tool lets you upload your plant pics.
    Image Display: We show you the uploaded image because, well, it’s nice to see what you just uploaded!
    Predict Button: Click it, and our model will tell you what’s going on with your plant. It's like having a plant doctor on call.

Wrapping It Up

And there you have it—a plant disease classifier that knows its stuff. We’ve built a neural network, set up image preprocessing, and wrapped it all up in a nice Streamlit interface.

Want to add more features? Go for it! Turn it into a mobile app or even add a “Plant Encouragement” feature that says something nice if your plant’s struggling. (Because who doesn’t love a little positive reinforcement?)

Happy coding, and may your plants never get the “Cedar apple rust”!
