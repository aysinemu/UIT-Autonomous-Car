# UIT Autonomous Car - Real-time Self-driving AI with YOLO and PID

This project demonstrates a comprehensive **autonomous driving pipeline** built for the UIT smart car platform. The system integrates:

* Real-time road-following using traditional **image processing** and **PID control**
* Smart behavior switching (turns, stops, etc.) using **YOLO object detection**
* Traffic sign interpretation with **semantic segmentation + logic control**

> This is the brain behind a self-driving vehicle built for road-simulation competitions at the **University of Information Technology (UIT)**.

## Key Features

* Raw camera input and semantic segmentation stream from simulator or car
* Lane detection using region-of-interest (ROI) and brightness thresholding
* PID-based steering control with dynamic deviation calculation
* Pre-trained **YOLO model** for traffic sign detection (Stop, Turn Left, Turn Right, etc.)
* Behavior switching logic to stop, turn, or go forward based on detected signs
* Time-based delay when stopping at Stop sign (realistic)
* Fully customizable logic flow with debugging printouts

## Project Structure

```
UIT-Autonomous-Car/
├── test_client.py               # Main loop, control logic, image processing
├── client_lib.py                # Library for car-server communication (GetRaw, AVControl, etc.)
├── /workspace/best.pt           # Pre-trained YOLOv8 model for sign recognition
└── README.md                    # Project documentation
```

## Control Logic

### Lane Following:

* Mid-lane point and second row are extracted from grayscale segmented image
* PID controller calculates deviation from center and outputs steering angle

### Object Detection:

* YOLO detects traffic signs from raw image stream
* If "TurnLeft", "TurnRight", "Stop", "NoTurnLeft", etc. is detected → switches mode

### Mode-Based Action:

| Mode | Action                |
| ---- | --------------------- |
| 0    | Normal lane following |
| 2    | Turn Left             |
| 3    | Turn Right            |
| 4    | Stop                  |
| 5    | Ignore TurnLeft       |
| 6    | Ignore TurnRight      |

### Speed Control:

* PID controller also used to maintain speed during turns or stops
* Stop mode triggers high PID values to bring vehicle to a halt for \~10s

## Dependencies

Install with:

```bash
pip install opencv-python numpy ultralytics
```

Model path:

* Place `best.pt` in `/workspace/` or adjust the path in:

```python
model = YOLO("/workspace/best.pt")
```

## How to Run

```bash
python test_client.py
```

Ensure your simulator or real vehicle is streaming the following:

* `GetRaw()` → returns raw RGB camera image
* `GetSeg()` → returns semantic segmentation image
* `GetStatus()` → returns current speed
* `AVControl(speed, angle)` → sets control commands to the car

Press `q` to quit.

## PID Controller Logic

```python
angle = P + I + D
```

Where:

* `P = error * Kp`
* `I = accumulated_error * Ki`
* `D = delta_error / delta_time * Kd`

Clamp output between \[-25, 25] degrees for safety.

## Example Debug Output

```
---------------------------------------
|  deviation = -14
|  frameleft = 0
|  frameright = 1
|  framestop = 0
|  frameredleft = 0
|  frameredright = 0
|  yolo = 1
|  timer = 7
|  forward = 1
---------------------------------------
```

## Notes

* `segment()` returns the detected YOLO class (e.g., "Stop", "TurnRight")
* YOLO model confidence threshold is set to `0.3`
* Uses `region_selection_road()` to mask irrelevant parts of the image
* Frame counters are used to ensure stable detection before switching mode

## Author

* **Nguyen Chau Tan Cuong** – University of Information Technology (HCMUTE)

> Developed as part of a competition or academic project for intelligent transportation systems.
