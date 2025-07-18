<!DOCTYPE html>
<html>

<body>
<h1>Aero Engine Defect Detection</h1>

<h2>Overview</h2>
<p>Computer vision system for detecting defects in aircraft engine components using Vision Transformer (ViT) architecture.</p>

<h2>Dataset</h2>
<ul>
<li>Training: 191 images</li>
<li>Validation: 52 images</li>
<li>Evaluation: 48 images</li>
</ul>

<h2>Defect Classes</h2>
<ul>
<li>0: Scratch</li>
<li>1: Dirty</li>
<li>2: Stain</li>
<li>3: Damage</li>
</ul>

<h2>Model</h2>
<p>Base Model: google/vit-base-patch16-224-in21k</p>
<p>Fine-tuned for 4-class defect classification</p>

<h2>Training Configuration</h2>
<ul>
<li>Epochs: 50</li>
<li>Learning Rate: 2e-5</li>
<li>Batch Size: 8</li>
<li>Image Size: 224x224</li>
<li>Weight Decay: 0.01</li>
</ul>

<h2>Results</h2>
<ul>
<li>Accuracy: 98.08%</li>
<li>Precision: 98.18%</li>
<li>Recall: 98.08%</li>
</ul>

<h3>Sample Results</h3>
<p>Classification results on test images:</p>
<p><img src="assets/result.png" alt="Sample Predictions" width="800"></p>

<h2>Training Loss</h2>
<ul>
<li>Initial Training Loss: 1.044</li>
<li>Final Training Loss: 0.019</li>
<li>Initial Validation Loss: 0.781</li>
<li>Final Validation Loss: 0.041</li>
</ul>

<h2>Technologies Used</h2>
<ul>
<li>PyTorch</li>
<li>Transformers</li>
<li>Vision Transformer (ViT)</li>
<li>Scikit-learn</li>
<li>Matplotlib</li>
<li>PIL</li>
<li>NumPy</li>
</ul>

<h2>Key Features</h2>
<ul>
<li>Custom dataset handler for image-label pairs</li>
<li>Data cleaning and preprocessing pipeline</li>
<li>Loss tracking and visualization</li>
<li>Model evaluation with multiple metrics</li>
<li>Prediction visualization</li>
</ul>

<h2>Usage</h2>
<pre>
# Load model
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224-in21k',
    num_labels=4
)
model.load_state_dict(torch.load('best_model.pth'))

# Preprocess image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# Make prediction
outputs = model(image_tensor)
prediction = torch.argmax(outputs.logits, dim=1)
</pre>

</body>
</html>
