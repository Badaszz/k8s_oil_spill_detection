# Oil Spill Detection

An AI-powered oil spill detection system using deep learning. This project uses a UNet model trained on satellite imagery to detect and segment oil spills in ocean environments. The model is packaged as an ONNX model and served via a FastAPI application that can be deployed on Kubernetes.

This project was built in order to detect oil spill in images, after the mask is created using the mdoel, the original image is returned, with a transparent red colour over the region detected as oil spills.

## Features

- **Automated Detection**: Uses a trained UNet neural network to detect oil spills from satellite imagery
- **REST API**: FastAPI-based REST endpoint for easy integration
- **ONNX Model**: Lightweight, cross-platform model format for efficient inference
- **Kubernetes Ready**: Includes deployment manifests for scalable cloud deployment
- **Docker Containerized**: Ready to deploy using Docker containers
- **Real-time Inference**: Accepts image URLs and returns segmentation masks with overlay visualization

## Project Structure

```
├── main.py                 # FastAPI application and inference logic
├── train.ipynb            # Jupyter notebook for model training
├── test.py                # Testing script to validate predictions
├── pyproject.toml         # Python project dependencies and configuration
├── Dockerfile             # Docker container definition
├── model/
│   └── unet_oilspill.onnx # Pre-trained UNet model in ONNX format
├── k8s/
│   ├── deployment.yaml    # Kubernetes deployment configuration
│   ├── service.yaml       # Kubernetes service configuration
│   └── hpa.yaml           # Horizontal Pod Autoscaler configuration
└── README.md              # This file
```

## Requirements

- Python >= 3.11
- FastAPI >= 0.126.0
- ONNX Runtime >= 1.23.2
- OpenCV (headless) >= 4.11.0.86
- Pillow >= 12.0.0
- Requests >= 2.32.5
- Uvicorn >= 0.40.0
- NumPy >= 2.4.0
- Matplotlib >= 3.10.8

## Installation

### Local Development

1. Clone the repository:
```bash
git clone <repository-url>
cd Oil_spill_detection
```

2. Install dependencies using UV (recommended):
```bash
uv sync
```

Or using pip:
```bash
pip install -r requirements.txt
```

### Docker

Build the Docker image:
```bash
docker build -t oil-spill-detection:latest .
```

Run the container:
```bash
docker run -p 8080:8080 oil-spill-detection:latest
```

## Usage

### API Endpoints

#### Predict
- **Endpoint**: `POST /predict`
- **Description**: Sends an image URL and receives an oil spill segmentation mask overlay
- **Request Body**:
```json
{
  "image_url": "https://example.com/image.jpg"
}
```
- **Response**: PNG image with oil spill regions highlighted in red

#### Health Check
- **Endpoint**: `GET /`
- **Description**: Basic endpoint for health checks

### Example Usage

```python
import requests
from PIL import Image
import io

url = "http://localhost:8080/predict"
request = {
    "image_url": "https://example.com/satellite_image.jpg"
}

response = requests.post(url, json=request)
img = Image.open(io.BytesIO(response.content))
img.show()
```

### Testing

Run the test script to validate predictions:
```bash
python test.py
```

## Model Details

- **Architecture**: U-Net Segmentation Network
- **Input**: 128x128 grayscale images
- **Output**: 128x128 binary segmentation mask
- **Format**: ONNX (Open Neural Network Exchange)
- **Preprocessing**: Images are normalized with mean=0.5 and std=0.5
- **Inference Threshold**: 0.5 (sigmoid activation)

## Training

To train or retrain the model, use the Jupyter notebook:
```bash
jupyter notebook train.ipynb
```

## Kubernetes Deployment

### Prerequisites

- Kubernetes cluster (tested on AWS EKS)
- Docker image pushed to a container registry

### Deploy

1. Update the image repository in `k8s/deployment.yaml`:
```yaml
image: <your-registry>/oil-spill-detection:latest
```

2. Configure subnet IDs in `k8s/service.yaml` for AWS Load Balancer

3. Apply the configuration:
```bash
kubectl apply -f k8s/
```

### Configuration

- **Replicas**: 2 (configurable in deployment.yaml)
- **CPU Request**: 300m, Limit: 500m
- **Memory Request**: 128Mi, Limit: 256Mi
- **Port**: 9696 (external), 8080 (internal)

## Performance Metrics

The model performance can be evaluated on test data. Refer to `train.ipynb` for validation metrics and evaluation procedures.

## Troubleshooting

- **Model Loading Issues**: Ensure the ONNX runtime is properly installed with CPU provider
- **Image Processing**: Verify image URLs are accessible and in supported formats (JPEG, PNG)
- **Kubernetes Health Checks**: Ensure the API responds to GET requests on the root path

