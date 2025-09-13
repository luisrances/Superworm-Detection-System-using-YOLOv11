# Superworm Detection using YOLOv11

This project implements a custom object detection model using Ultralytics YOLOv11 to automatically detect and count superworms in images. It is part of a thesis work on computer vision applications in biological specimen analysis.

## Features

- Custom YOLOv11 model training for superworm detection
- Automated counting of superworms in images
- Model validation and performance visualization
- Dataset management using Roboflow
- Training results analysis with loss metrics and mAP scores

## Technologies Used

- **Python**: Programming language
- **Ultralytics YOLOv11**: Deep learning framework for object detection
- **Roboflow**: Dataset annotation and management platform
- **Pandas**: Data manipulation and analysis
- **Matplotlib**: Data visualization
- **Google Colab**: Cloud-based training environment

## Dataset

The dataset consists of annotated images of superworms, split into training, validation, and test sets. The dataset is configured in `dataset_custom.yaml` with:
- Training images: `train/`
- Validation images: `valid/`
- Number of classes: 1 (superworms)

## Installation

1. Create a conda environment:
   ```bash
   conda create -n superworm_detection python=3.8
   conda activate superworm_detection
   ```

2. Install required packages:
   ```bash
   pip install ultralytics roboflow pandas matplotlib
   ```

3. Verify CUDA availability (optional, for GPU training):
   ```python
   import torch
   print(torch.cuda.is_available())
   ```

## Usage

### Training

Due to hardware limitations, training is recommended on Google Colab. Use the provided `colab training.ipynb` notebook:

1. Upload the notebook to Google Colab
2. Install dependencies
3. Download dataset from Roboflow
4. Run training with 300 epochs

For local training (CPU):
```python
from ultralytics import YOLO

model = YOLO("yolo11s.pt")
results = model.train(
    data="dataset_custom.yaml",
    epochs=2,  # Increase for better performance
    imgsz=640,
    device="cpu"
)
```

### Validation

```python
from ultralytics import YOLO

model = YOLO("trained_model.pt")
model.val(data="dataset_custom.yaml")
```

### Prediction

```python
from ultralytics import YOLO

model = YOLO("trained_model.pt")
results = model.predict(source="path/to/image.jpg", conf=0.6, show_labels=True)

# Count detected superworms
total_superworms = sum(r.boxes.data.shape[0] for r in results)
print(f"Detected {total_superworms} superworms")
```

## Results

Training results are saved in `runs/detect/train/` including:
- Loss curves (box_loss, cls_loss, dfl_loss)
- mAP metrics (mAP50, mAP50-95)
- Confusion matrices
- Precision-Recall curves

Use `final_py.py` to visualize training results and perform predictions.

## Project Structure

```
yolov_11_custom/
├── .gitignore
├── README.md
├── colab training.ipynb      # Colab training notebook
├── final_py.py               # Main script for validation and prediction
├── dataset_custom.yaml       # Dataset configuration
├── trained_model.pt          # Final trained model
├── yolo11s.pt               # Pre-trained YOLOv11 model
├── process/                  # Training and utility scripts
├── runs/                     # Training results and visualizations
├── train/                    # Training dataset
├── valid/                    # Validation dataset
└── test/                     # Test dataset
```

## Contributing

This is a thesis project. For contributions or questions, please contact the project maintainer.

## License

This project is for educational and research purposes.
