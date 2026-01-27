# birdnet-mic-stream

Detect bird species in real-time using microphone audio input and the BirdNET deep learning model.


## Goals

- Enable uninterrupted, real-time detection of bird species from live microphone input.
- Processing audio streams in-memory without temporary file creation.
- Reduce storage operations and disk access to preserve system resources and extend hardware lifespan.
- Target Raspberry Pi as the primary deployment platform with efficient resource utilization.
- Provide export capabilities to send detection data to the Cloud for futher processing and analytics.

## Installation

### Setup

1. **Create a virtual environment (recommended):**
```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate     # On Windows
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Usage

Run bird monitoring that detects birds in real-time:
```bash
# Start monitoring with default settings
python bird_monitor.py

# Start monitoring with 1 minute chunk duration and 70% confidence threshold
python bird_monitor.py --duration 60 --confidence 0.7

# Display detailed usage instructions
python bird_monitor.py -h
```

**Stop monitoring:** Press `Ctrl+C` to stop

## About BirdNET

Developed by the [K. Lisa Yang Center for Conservation Bioacoustics](https://www.birds.cornell.edu/ccb/) at the [Cornell Lab of Ornithology](https://www.birds.cornell.edu/home) in collaboration with [Chemnitz University of Technology](https://www.tu-chemnitz.de/index.html.en).

Go to https://birdnet.cornell.edu to learn more about the project.

`birdnet-mic-stream` is not associated with BirdNET or the K. Lisa Yang Center for Conservation Bioacoustics.
