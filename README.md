# MNIST Handwritten Digit Classifier (PyTorch + CNN)

This project is a simple Convolutional Neural Network (CNN) built using PyTorch.  
It trains on the MNIST dataset to classify handwritten digits (0–9).  
The project includes data loading, training, saving/loading the model, and predicting from custom images.

---

## Project Structure

```
├── data/               # MNIST dataset (auto-downloaded)
├── model_state.pt      # Saved trained model
├── img_3.jpg           # Example image for prediction
├── torchnn.py             # Training + inference script
└── README.md           # This file
```

---

## Features

- Three convolutional layers for feature extraction  
- GPU acceleration using CUDA  
- Saves and loads model weights through state_dict  
- Can predict handwritten digits from custom images  

---

## Requirements

Install the necessary libraries:

```bash
pip install torch torchvision pillow
```

Ensure you have a CUDA-enabled version of PyTorch if using a GPU.

---

## Training the Model

Run the script:

```bash
python torchnn.py
```

The training loop runs for 10 epochs using:
- Adam optimizer
- CrossEntropyLoss
- Batch size of 32
- CUDA for training (if available)

After training, the model is automatically saved to:

```
model_state.pt
```

---

## Predicting Using a Custom Image

Place your image inside the project directory (example: `img_3.jpg`).  
Make sure it is a grayscale, 28x28 handwritten digit similar to MNIST.

Prediction happens at the bottom of `main.py`:

```python
img = Image.open('img_3.jpg')
img_tensor = ToTensor()(img).unsqueeze(0).to('cuda')
print(torch.argmax(clf(img_tensor)))
```

This prints the predicted digit.

---

## Notes

- The model expects a grayscale image (1 channel).
- Image size should be 28x28.
- If you run this on a machine without CUDA, change `'cuda'` to `'cpu'`.

---

## License

This project is open-source and free to use.
