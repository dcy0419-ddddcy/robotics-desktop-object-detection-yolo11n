# Desktop Object Detection and Real-Time Recognition with YOLO11n

Course submission for **Robotics Integration Group Project - Experiment 1: Object Detection and Recognition**.

- Student: Chengyang Deng
- Student ID: 24020036012
- Class: Class 1, Grade 2024
- Instructor: Xiaowei Zhou
- University: Ocean University of China

## Submission contents

| Directory | Required material | Contents |
|---|---|---|
| `01_report` | Experiment report | Final compiled PDF only |
| `02_dataset` | Dataset | Four-class YOLO dataset, split configuration, manifest, and summary |
| `03_model` | Model | Best PyTorch checkpoint and fixed-shape ONNX export |
| `04_source_code` | Programs and run instructions | Data preparation, training, evaluation, error analysis, camera, Jetson, and ROS 2 programs |
| `05_results` | Saved test results and error cases | Training records, evaluation plots, annotated predictions, and representative failures |
| `06_video` | Result video | Camera-only real-time detection recording |

## Experiment summary

The detector recognizes four desktop-object classes: `mouse`, `keyboard`, `laptop`, and `headphones`. The final dataset contains 1,343 images and 1,878 bounding boxes. The independent test split contains 123 images and 213 annotated objects. At confidence 0.25 and matching IoU 0.5, 174 objects are matched correctly, corresponding to an 81.69% recognition rate.

The selected model is YOLO11n. Its independent-test metrics are precision 0.882, recall 0.762, mAP@0.5 0.810, and mAP@0.5:0.95 0.564. The Windows camera demonstration displays class names, bounding boxes, confidence scores, and FPS. Core programs for Jetson benchmarking and ROS 2 JSON publication are included in `04_source_code`.

## Main entry points

Run the programs in this order when rebuilding the experiment:

1. `prepare_data.py`
2. `train.py`
3. `evaluate.py`
4. `analyze_errors.py`
5. `export_onnx.py`

Run `camera_detect.py` separately for Windows USB-camera detection. On Jetson, use `jetson_benchmark.py` for the fixed 20-frame warm-up and 200-frame benchmark, then use `ros2_detector_node.py` and `ros2_result_subscriber.py` to publish and verify `/object_detection/result`.

Detailed commands are provided in `04_source_code/README.md` and in the report appendix.

## Key files

- Report: `01_report/experiment_report.pdf`
- Dataset configuration: `02_dataset/four_class_dataset/data.yaml`
- PyTorch model: `03_model/best.pt`
- ONNX model: `03_model/best.onnx`
- Training metrics: `05_results/training/results.csv`
- Error summary: `05_results/error_analysis/ERROR_CASES.csv`
- Result video: `06_video/result_video_camera_only.mp4`

LaTeX source, virtual environments, editor settings, cache files, intermediate datasets, and obsolete five-class results are intentionally excluded from this submission repository.
