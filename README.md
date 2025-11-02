# TrafficVision: AI-Powered Traffic Image Assessment Platform 🚗🚦

TrafficVision is a cross-platform desktop application designed for AI-powered traffic safety analysis and urban planning. It utilizes a deep learning model to accurately detect and classify key traffic objects from static images.

The system adheres to a three-tier architecture, featuring a FastAPI server for managing the AI processing pipeline and a user-friendly Flet client for the graphical interface, allowing for efficient, automated safety assessments.

## ✨ Key Features and Functions

The TrafficVision system provides comprehensive tools for traffic safety assessment and data analysis:

- **Core Object Classification**: The system detects and classifies eight critical traffic object types: person, bicycle, car, motorcycle, bus, truck, traffic light, and stop sign.
- **Batch Processing**: Supports processing multiple traffic images sequentially for bulk analysis and automated summary report generation.
- **Adjustable Threshold**: Allows users to configure the detection confidence threshold in real-time to adjust detection sensitivity.
- **Analysis History**: Maintains a chronological history log of all completed single and batch analyses.
- **Multi-Format Export**: Provides comprehensive export functionality supporting PDF, CSV, and JSON file formats for data sharing and detailed reporting.

## 🧠 AI Model and Architecture

### Model Details

The system's core object detection capability is built using Transfer Learning.

- **Architecture**: EfficientNet-B1
- **Training Foundation**: Pre-trained on the COCO2017 subset
- **Performance Goal**: The system is designed to achieve a minimum of 90% mean Average Precision (mAP) and maintain processing times under 5 seconds per image.
- **Model File**: The checkpoint file `efficientnet_b1_8class_multilabel_BEST.ckpt` is required in the `server/` directory.

### Technology Stack

| Component | Tier | Technology | Role |
|-----------|------|------------|------|
| Client | Presentation | Flet | Provides the cross-platform, pure-Python desktop interface |
| Server | Logic | FastAPI | Asynchronous API for managing model inference and business logic |
| Database | Data | SQLite | Storage for analysis history, detection results, and user preferences |

## 💻 Installation and Setup

### 1. Clone Repository and Install Dependencies

```bash
# Clone the repository
git clone https://github.com/allendalangin/TrafficVision.git
cd TrafficVisionProject

# Install required Python packages
pip install flet fastapi uvicorn python-multipart pydantic httpx mindspore mindcv fpdf
```

## 2. File Setup

Ensure the model checkpoint is correctly placed:
- Place `efficientnet_b1_8class_multilabel_BEST.ckpt` inside the `server/` directory.

## ▶️ Running the Application

The system uses a client-server architecture and requires two separate processes to run concurrently.

### 1. Start the FastAPI Server (Backend)

Open your first terminal in the project root directory and run the server using uvicorn. This command enables hot-reloading (`--reload`) for development:

```bash
uvicorn server.api_server:app --reload --host 127.0.0.1 --port 8000
```

### 2. Start the Flet Client (Frontend)

Open a new terminal window in the project root directory and launch the desktop application. The Flet client will automatically attempt to connect to the running FastAPI server:

```bash
flet run main.py
```
