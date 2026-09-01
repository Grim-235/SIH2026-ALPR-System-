# Modern Indian ALPR 🚗

![License](https://img.shields.io/badge/license-Apache%202.0-blue)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![Status](https://img.shields.io/badge/status-Production%20Ready-brightgreen)

**Modern Indian ALPR** is a comprehensive, production-ready Automatic License Plate Recognition (ALPR) system designed specifically for Indian vehicle license plates. Built with cutting-edge deep learning and computer vision technologies, it provides real-time plate detection, character recognition, and traffic analytics capabilities.

## 🎯 Key Features

- **🎯 High-Accuracy Plate Detection** - YOLOv8-based detector optimized for Indian license plates
- **📝 Robust OCR Recognition** - EasyOCR for accurate text extraction from plates
- **🎬 Multi-Source Support** - Process images, videos, folders, or live webcam feeds
- **📊 Advanced Video Tracking** - Track vehicles across frames with trajectory analysis
- **🌐 Web Dashboard** - Flask-based REST API with interactive frontend
- **📈 Analytics Dashboard** - Streamlit-powered real-time traffic analytics
- **💾 SQLite Database** - Persistent storage for detections, alerts, and blacklists
- **🚀 GPU Acceleration** - CUDA support for faster inference (fallback to CPU)
- **📱 Easy Configuration** - JSON-based camera and blacklist management
- **🔄 Batch Processing** - Process multiple images/videos efficiently

## 📋 System Requirements

### Minimum Requirements
- **OS**: Windows, Linux, or macOS
- **Python**: 3.10 - 3.12 (3.14+ works but may need CPU-only PyTorch)
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 5GB for models and dependencies

### Optional GPU Support
- **NVIDIA GPU**: RTX series or better
- **CUDA**: 11.8 or 12.1 compatible with your GPU
- **cuDNN**: 8.x series

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Grim-235/SIH2026-ALPR-System-.git
cd SIH2026-ALPR-System-
```

### 2. Create Virtual Environment
```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# Linux/macOS
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 3. Install PyTorch

**Option A: CPU-Only (Quick, works everywhere)**
```powershell
python -m pip install torch torchvision torchaudio
```

**Option B: GPU Support (NVIDIA RTX cards)**
```powershell
# For CUDA 11.8
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install Dependencies
```powershell
python -m pip install -r requirements.txt
```

### 5. Run a Quick Test
```powershell
# Process an image (model auto-downloads on first run)
python modern_alpr.py --source inputs/1.jpg --output results/output.jpg

# Process a video
python modern_alpr.py --source path/to/video.mp4 --output results/ --show

# Use webcam (live preview)
python modern_alpr.py --source 0 --show
```

## 📖 Usage Guide

### Command-Line Interface (modern_alpr.py)

The most direct way to use ALPR on images, videos, or webcam feeds.

```powershell
python modern_alpr.py [OPTIONS]
```

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--source` | `inputs/1.jpg` | Image/video path, folder, or webcam index (0, 1, etc.) |
| `--output` | `results/modern_output.jpg` | Output file path or folder |
| `--model` | `data/models/license_plate_yolov8_best.pt` | YOLO detector .pt file path |
| `--download-model` | - | Auto-download model if missing |
| `--conf` | `0.35` | Minimum detection confidence (0-1) |
| `--iou` | `0.5` | YOLO NMS IoU threshold |
| `--imgsz` | `640` | YOLO inference image size |
| `--device` | `auto` | Device: auto, cpu, cuda, cuda:0 |
| `--ocr` | `easyocr` | OCR backend: easyocr, none |
| `--show` | - | Display live preview window |
| `--max-frames` | `0` | Stop after N frames (0 = no limit) |

**Examples:**
```powershell
# Detect plates in a single image
python modern_alpr.py --source image.jpg --output results/image_detected.jpg --show

# Process entire folder
python modern_alpr.py --source inputs/ --output results/batch/ --device cuda

# Live webcam detection
python modern_alpr.py --source 0 --show --device cuda

# Video processing with lower confidence threshold
python modern_alpr.py --source video.mp4 --output results/ --conf 0.3 --show

# Process first 100 frames of video
python modern_alpr.py --source video.mp4 --output results/ --max-frames 100
```

### Web Dashboard (app.py)

Full-featured REST API with interactive web interface for fleet management and traffic analytics.

```powershell
python app.py
# Open browser: http://localhost:5000
```

**Features:**
- Real-time vehicle detection and tracking
- Blacklist management for stolen/wanted vehicles
- Detection history and statistics
- Heatmap visualization of traffic patterns
- Alert system for blacklisted plates
- Multi-camera support via cameras.json

**API Endpoints:**
- `GET /` - Web interface
- `POST /api/process-video` - Upload and process video
- `GET /api/stats` - Get detection statistics
- `GET /api/detections` - Get recent detections
- `GET /api/blacklist` - Fetch current blacklist
- `POST /api/blacklist/add` - Add to blacklist
- `DELETE /api/blacklist/remove` - Remove from blacklist

### Analytics Dashboard (dashboard.py)

Real-time traffic analytics and reporting with interactive visualizations.

```powershell
streamlit run dashboard.py
# Opens browser automatically: http://localhost:8501
```

**Dashboard Features:**
- Live detection statistics
- Traffic pattern analysis
- Vehicle route tracking
- Peak hour analytics
- Blacklist alerts visualization
- Camera-wise performance metrics

## 📁 Project Structure

```
Modern-Indian-ALPR/
├── modern_alpr.py           # CLI tool for batch ALPR processing
├── app.py                   # Flask web backend & REST API
├── dashboard.py             # Streamlit analytics dashboard
├── worker.py                # Background job worker
├── simulate_cameras.py      # Camera simulation for testing
├── process_videos.py        # Batch video processor
├── trajectory.py            # Vehicle trajectory analysis
│
├── alpr/                    # Core ALPR module
│   ├── detector.py         # YOLOv8 plate detection
│   ├── ocr.py              # EasyOCR text recognition
│   ├── tracker.py          # Vehicle tracking across frames
│   ├── database.py         # SQLite database operations
│   └── __init__.py
│
├── data/                    # Data directory
│   ├── models/
│   │   └── license_plate_yolov8_best.pt    # YOLOv8 detector
│   ├── logs/               # Processing logs
│   └── uploads/            # Temporary file uploads
│
├── templates/              # Web interface templates
│   └── index.html
├── static/                 # Frontend assets
│   ├── app.js
│   └── styles.css
│
├── inputs/                 # Input images/videos for processing
├── results/                # Output results (images, videos, reports)
│
├── cameras.json            # Camera configuration
├── blacklist.txt           # Blacklisted license plates
├── requirements.txt        # Python dependencies
└── LICENSE                 # Apache 2.0 License
```

## 🔧 Configuration

### Camera Configuration (cameras.json)
Define multiple camera sources and their metadata:

```json
{
  "cameras": [
    {
      "id": "cam_001",
      "name": "Main Highway - North Gate",
      "location": "12.9716,77.5946",
      "source": "rtsp://camera_ip/stream",
      "enabled": true
    },
    {
      "id": "cam_002",
      "name": "City Center",
      "location": "12.9352,77.6245",
      "source": "inputs/sample_video.mp4",
      "enabled": true
    }
  ]
}
```

### Blacklist Management (blacklist.txt)
Store known stolen or wanted vehicle plates:

```
KA-01-AB-1234
DL-10-XY-5678
MH-02-CD-9101
```

Or add dynamically via the web dashboard API:
```bash
curl -X POST http://localhost:5000/api/blacklist/add \
  -H "Content-Type: application/json" \
  -d '{"plate": "KA-01-AB-1234", "reason": "Stolen vehicle"}'
```

## 📊 Output Formats

### Detection JSON Output
```json
{
  "frame": 125,
  "detections": [
    {
      "box": [150, 200, 250, 280],
      "detection_confidence": 0.94,
      "text": "KA-01-AB-1234",
      "ocr_confidence": 0.92,
      "matches_indian_plate_pattern": true
    }
  ]
}
```

### Heatmap Visualization
- Geographic heatmaps of detection density
- Traffic flow analysis
- Peak hour visualization

### Trajectory Analysis
- Vehicle path tracking across cameras
- Route frequency analysis
- Anomaly detection

## 🛠️ Troubleshooting

### Issue: Model download fails
```powershell
# Manually download from Hugging Face
# https://huggingface.co/ddeep/license_plate_yolov8_best

# Or specify local model:
python modern_alpr.py --model data/models/your_model.pt --source image.jpg
```

### Issue: Out of memory
```powershell
# Reduce image size for inference
python modern_alpr.py --imgsz 480 --source video.mp4

# Use CPU instead of GPU
python modern_alpr.py --device cpu --source video.mp4
```

### Issue: Poor detection accuracy
- Adjust confidence threshold: `--conf 0.25` (lower = more detections)
- Ensure adequate lighting in source video/images
- Train custom model with your specific plate types

### Issue: GPU not detected
```powershell
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Force CPU usage
python modern_alpr.py --device cpu
```

## 🎓 How It Works

### 1. **Plate Detection (YOLOv8)**
- Input: Image/video frame
- Process: YOLO object detection
- Output: Bounding box coordinates with confidence scores
- Model: Lightweight YOLOv8n variant for real-time performance

### 2. **Text Recognition (EasyOCR)**
- Input: Cropped license plate region
- Process: Deep learning-based OCR
- Output: Recognized text with character-level confidence
- Validation: Indian plate pattern matching (e.g., KA-01-AB-1234)

### 3. **Tracking (Multi-Object Tracking)**
- Input: Frame-by-frame detections
- Process: Associate detections across frames
- Output: Vehicle trajectories and routes
- Use: Traffic analysis and vehicle counting

### 4. **Database Storage**
- Input: Detection results
- Process: Schema-normalized insertion
- Output: Queryable detection history
- Features: Blacklist alerts, time-series analytics

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

The included YOLOv8 model is Apache-2.0 licensed.

## 📚 References & Resources

- **YOLOv8 Documentation**: https://docs.ultralytics.com/
- **EasyOCR**: https://github.com/JaidedAI/EasyOCR
- **OpenCV**: https://opencv.org/
- **Flask**: https://flask.palletsprojects.com/
- **Streamlit**: https://streamlit.io/
- **PyTorch**: https://pytorch.org/

## 🐛 Known Limitations

- Optimized for Indian license plates; may need fine-tuning for other formats
- Real-time processing speed depends on hardware (CPU vs GPU)
- Requires internet connection for initial model download
- SQLite database suitable for single-machine deployments (consider PostgreSQL for scale)

## 📞 Support & Contact

- **GitHub Issues**: Report bugs and request features
- **Discussions**: Ask questions and share ideas
- **Email**: Check repository for maintainer contact

## 🎯 Roadmap

- [ ] Multi-GPU support for distributed processing
- [ ] REST API authentication and rate limiting
- [ ] Mobile app for real-time alerts
- [ ] Custom model training pipeline
- [ ] Integration with traffic management systems
- [ ] Advanced pattern recognition for vehicle type detection

## ⭐ Acknowledgments

Built with ❤️ for the Smart India Hackathon 2026

---

**Happy detecting! 🚗📹**

If you found this project helpful, please give it a ⭐ star!

The first OCR run downloads EasyOCR model files into `.cache/easyocr`. The first detector run can download `data/models/license_plate_yolov8_best.pt`.

## Run

Download the default detector and process one image:

```powershell
python modern_alpr.py --download-model --source inputs/1.jpg --output results/modern_output.jpg
```

Process all sample images:

```powershell
python modern_alpr.py --source inputs --output results/modern_batch
```

Process video:

```powershell
python modern_alpr.py --source inputs/demo1.mp4 --output results/modern_demo1.mp4
```

Use webcam:

```powershell
python modern_alpr.py --source 0 --output results/webcam_modern.mp4 --show
```

Detection-only demo if OCR dependencies are slow or unavailable:

```powershell
python modern_alpr.py --source inputs/1.jpg --output results/detection_only.jpg --ocr none
```

## Accuracy Notes

The included default model is for license plate localization, not Indian-plate-specific OCR. For the strongest hackathon result, collect 100-300 images from your expected camera/device angle and fine-tune a YOLO detector on Indian plates. Roboflow Universe has Indian plate datasets and trained model APIs that can help bootstrap this.

The OCR cleanup accepts common Indian formats such as `MH12AB1234` and Bharat-series plates such as `22BH1234AA`.

## Tested Locally

This setup was tested on Windows with Python 3.14.3 and CPU PyTorch:

```powershell
.\.venv\Scripts\python.exe modern_alpr.py --source inputs --output results\modern_batch
.\.venv\Scripts\python.exe modern_alpr.py --source inputs\demo1.mp4 --output results\modern_demo1_short.mp4 --ocr none --max-frames 20
```

The sample image batch detected one plate in each of the four test images and produced these OCR reads:

- `DL2CAY3180`
- `MH12JC2813`
- `TS08FM8888`
- `MH14DX5842`

## References

- Ultralytics YOLO Python predict mode: https://docs.ultralytics.com/modes/predict/
- EasyOCR usage and Windows install notes: https://github.com/JaidedAI/EasyOCR
- Default plate detector model: https://huggingface.co/yasirfaizahmed/license-plate-object-detection
