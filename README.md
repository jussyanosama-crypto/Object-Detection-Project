# YOLOv11 Terminal & Streamlit Deployment

A production-ready scaffold for deploying YOLOv11 models locally and on Streamlit Cloud.

## 📁 Folder Structure

```
yolo-terminal-deployment/
│
├── app.py                 # Local terminal testing script
├── streamlit_app.py       # Streamlit web application
├── requirements.txt       # Python dependencies
├── packages.txt           # System dependencies (for Streamlit Cloud)
├── README.md              # Project documentation
│
├── model/                 # Model files
│   ├── best.pt            # Your trained YOLOv11 model
│   └── labels.txt         # Class labels (one per line)
│
├── utils/                 # Core functionality
│   ├── detector.py        # Model loading and inference logic
│   └── visualization.py   # Bounding box drawing utilities
│
└── assets/                # Static files
    └── demo.png           # Demo image for testing
```

## 🚀 How to Run Locally

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Add your model:**
   Place your trained model at `model/best.pt` and your classes in `model/labels.txt`.

3. **Terminal Test:**
   Run the terminal script to verify everything works:
   ```bash
   python app.py
   ```
   *This will output `output.jpg` with detections.*

4. **Web App:**
   Start the Streamlit application:
   ```bash
   streamlit run streamlit_app.py
   ```

## ☁️ Streamlit Cloud Deployment

1. Push this repository to GitHub.
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click "New app" and select your repository.
4. Set the main file path to `streamlit_app.py`.
5. Click "Deploy".

*Note: The `packages.txt` file automatically handles the system-level `libgl1` dependency required by OpenCV in Linux environments.*
