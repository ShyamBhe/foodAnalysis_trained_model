# FoodVision AI – Food Image Classification System

FoodVision AI is a deep-learning-based food image classification application that identifies food categories from uploaded images using a fine-tuned **MobileNetV3-Small** model trained on the Food-101 dataset.

The project combines PyTorch, FastAPI, React and Vite to provide an interactive web application for food image recognition.

## System Architecture

The application follows a frontend–backend architecture, separating the user interface from AI model inference.

**Workflow:**

1. The user uploads a food image through the React frontend.
2. The frontend sends the image to the FastAPI backend.
3. The backend preprocesses the image and passes it to the trained PyTorch model.
4. MobileNetV3-Small predicts the most likely food categories.
5. The backend returns the top five predictions and confidence scores.
6. The frontend displays the prediction results.

**Deployment architecture:**
- **Frontend:** Netlify – React + Vite
- **Backend:** Render – FastAPI + Python
- **AI model:** PyTorch + MobileNetV3-Small
- **Dataset:** Food-101

## Technologies

| Component | Technology |
|---|---|
| Programming language | Python, JavaScript |
| Deep learning | PyTorch |
| Computer vision | Torchvision, Pillow |
| AI architecture | MobileNetV3-Small |
| Backend | FastAPI |
| Frontend | React |
| Build tool | Vite |
| API communication | REST API |
| Model acceleration | Apple Metal Performance Shaders (MPS), where available |
| Deployment | Netlify and Render |

## Model and Dataset

### Food-101 Dataset

FoodVision AI uses the Food-101 dataset, a publicly available food image dataset containing:

- 101 food categories
- 101,000 images
- 750 training images per category
- 250 testing images per category

The dataset contains real-world food images with variations in appearance, presentation, lighting and background.

Dataset source: [Food-101 – ETH Zurich](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)

### Dataset Preparation

The official training dataset is divided into training and validation subsets using a 90/10 split.

| Dataset | Number of images |
|---|---:|
| Training | 68,175 |
| Validation | 7,575 |
| Official testing | 25,250 |
| Total | 101,000 |

The official test split is retained for final model evaluation.

### Data Preprocessing

Training transformations include:
- Random resized cropping
- Random horizontal flipping
- Image tensor conversion
- ImageNet normalization

Validation and testing transformations include:
- Image resizing
- Center cropping
- Image tensor conversion
- ImageNet normalization

## Deep Learning Model

The project uses **MobileNetV3-Small**, a lightweight convolutional neural network architecture designed for efficient image classification.

The model is initialized with ImageNet-pretrained weights and adapted to classify 101 food categories.

### Transfer Learning Strategy

1. Load the pretrained MobileNetV3-Small architecture.
2. Replace the original classification layer with a new 101-class output layer.
3. Freeze the feature extraction backbone and train the classification head.
4. Fine-tune the final feature blocks to improve food classification performance.
5. Save the best-performing model checkpoint based on validation accuracy.

### Model Performance

Initial evaluation of the baseline model on the official test dataset:

| Metric | Result |
|---|---:|
| Test accuracy | 61.66% |
| Macro precision | 63.42% |
| Macro recall | 61.66% |
| Macro F1-score | 61.08% |

The fine-tuned model achieved **62.35% validation accuracy** during training.

Note: The fine-tuned model's official test performance should be reported separately after evaluation on the untouched test dataset.

## Project Structure

```text
food-ai-model/
│
├── data/
│   └── food-101/
│       ├── images/
│       └── meta/
│
├── models/
│   ├── foodvision_baseline.pth
│   ├── foodvision_best.pth
│   ├── foodvision_finetuned_best.pth
│   └── class_names.json
│
├── src/
│   ├── data.py
│   ├── model.py
│   └── predict.py
│
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── App.css
│   │   └── index.css
│   ├── package.json
│   └── index.html
│
├── api.py
├── .gitignore
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd food-ai-model
```

### 2. Set up the Python environment

Python 3.11 is recommended.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Install backend dependencies

```bash
python -m pip install torch torchvision
python -m pip install fastapi "uvicorn[standard]" python-multipart pillow
```

For Apple Silicon Macs, PyTorch can use Metal Performance Shaders (MPS) when supported.

### 4. Prepare the dataset

Download Food-101 from the official ETH Zurich dataset page and extract it into:

```text
data/food-101/
```

The expected dataset structure is:

```text
data/food-101/
├── images/
│   ├── apple_pie/
│   ├── baby_back_ribs/
│   ├── baklava/
│   └── ...
└── meta/
```

### 5. Run the AI prediction script

```bash
python src/predict.py test.jpeg
```

### 6. Start the FastAPI backend

From the project root:

```bash
python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

Backend URL:

```text
http://127.0.0.1:8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

Health check:

```text
http://127.0.0.1:8000/health
```

### 7. Start the React frontend

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the local development URL displayed by Vite, usually:

```text
http://localhost:5173
```

## API Endpoints

The FastAPI backend provides the following endpoints:

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API information |
| GET | `/health` | Backend and model health |
| POST | `/predict` | Upload an image and receive predictions |

### Prediction request

The `/predict` endpoint accepts an image upload using multipart/form-data.

Supported formats:
- JPEG
- PNG
- WEBP

Maximum upload size: 10 MB.

Example response:

```json
{
  "filename": "food-image.jpeg",
  "predictions": [
    {
      "food": "Spaghetti Bolognese",
      "confidence": 92.45
    },
    {
      "food": "Lasagna",
      "confidence": 3.21
    }
  ],
  "device": "mps"
}
```

The response illustrates the top predictions; actual results depend on the uploaded image.

## Deployment

The planned deployment separates the frontend and backend.

### Frontend – Netlify

The React/Vite application can be deployed through a GitHub repository connection.

Typical build configuration:

```text
Base directory: frontend
Build command: npm run build
Publish directory: frontend/dist
```

### Backend – Render

The FastAPI application can be deployed as a Python web service or Docker-based service.

Start command:

```bash
uvicorn api:app --host 0.0.0.0 --port $PORT
```

The backend deployment must include the trained model checkpoint and class-name file.

Configure the frontend to use the public Render API URL rather than the local development address. Configure CORS to allow the deployed Netlify domain.

**Note:** Hosting requirements should be tested against the memory usage of PyTorch and the loaded model. Free hosting plans may not provide sufficient resources.

## Current Limitations and Future Improvements

- Improve classification accuracy through additional fine-tuning and experimentation.
- Evaluate the fine-tuned model on the official test dataset.
- Explore larger and more advanced vision architectures.
- Add confidence-based handling for uncertain predictions and images outside the 101 food categories.
- Integrate a nutritional database to estimate calories and macronutrients using predicted food categories and appropriate portion-size information.
- Explore multimodal AI models for more detailed food recognition and nutritional analysis.
- Optimize model inference and deployment performance.

## Dataset License and Attribution

The Food-101 dataset consists of images collected from Foodspotting. These images are not owned by ETH Zurich.

According to the original dataset license agreement, use beyond scientific fair use must be negotiated with the respective image owners in accordance with Foodspotting's terms of use.

Original dataset references:

1. [Foodspotting](http://www.foodspotting.com/)
2. [Foodspotting Terms of Use](http://www.foodspotting.com/terms/)

Dataset research reference:

Bossard, L., Guillaumin, M., and Van Gool, L. (2014). *Food-101 – Mining Discriminative Components with Random Forests*. European Conference on Computer Vision (ECCV).

## Author

**Shyam Bhetuwal**

AI & Software Engineer | Machine Learning | Deep Learning | Computer Vision

Finland

---

*FoodVision AI – Exploring practical applications of deep learning in food image recognition.*