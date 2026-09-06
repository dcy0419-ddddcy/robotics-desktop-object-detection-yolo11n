# 核心训练代码

本目录只保留九个核心程序：

- `prepare_data.py`：合并原数据与 Open Images V7 数据，删除水杯并生成四分类数据集。
- `train.py`：使用 YOLO11n 训练鼠标、键盘、笔记本电脑和耳机四类模型。
- `evaluate.py`：在独立测试集上评估，并保存带置信度的预测结果。
- `analyze_errors.py`：统计测试集漏检/误检，并保存典型错误案例。
- `export_onnx.py`：把最佳权重导出为 Jetson 可继续转换的 ONNX 模型。
- `camera_detect.py`：调用外置 USB 摄像头，实时显示四类检测框、置信度和 FPS。
- `jetson_benchmark.py`：在 Jetson 上预热 20 帧并测量 200 帧，保存逐帧测速 CSV。
- `ros2_detector_node.py`：在 Jetson 上运行摄像头检测，并向 `/object_detection/result` 发布 JSON 结果。
- `ros2_result_subscriber.py`：订阅并打印类别、置信度和节点 FPS，用于验证 ROS 2 通信。

最终四分类数据集已放在 `02_dataset/four_class_dataset`，可以直接运行 `train.py`、`evaluate.py`、`analyze_errors.py` 和 `export_onnx.py`。新生成的结果保存在 `05_results`。`prepare_data.py` 保留数据整理过程，但重新执行它还需要原始采集与公开数据源。

摄像头测试单独运行 `camera_detect.py`，它从 `03_model/best.pt` 加载模型；按 `Q` 退出，Windows 外置摄像头编号为 1。

## Jetson 与 ROS 2

把 `best.pt` 和三个 Jetson/ROS 2 程序复制到 Jetson，例如放在 `~/desktop_detector`，并将权重保存为 `~/desktop_detector/models/best.pt`。以下命令使用 PyTorch FP16，与同组已有 Jetson 验证方法保持一致：

```bash
cd ~/desktop_detector
python3 jetson_benchmark.py --model models/best.pt --source /dev/video0 \
  --device 0 --half --warmup 20 --frames 200
```

启动 ROS 2 发布节点：

```bash
source /opt/ros/humble/setup.bash
cd ~/desktop_detector
python3 ros2_detector_node.py --ros-args \
  -p model:=models/best.pt -p source:=/dev/video0 \
  -p device:=0 -p half:=true -p display:=true
```

另开终端验证消息内容与发布频率：

```bash
source /opt/ros/humble/setup.bash
ros2 topic echo /object_detection/result --once
ros2 topic hz /object_detection/result --window 200
python3 ~/desktop_detector/ros2_result_subscriber.py
```

测速 CSV 默认保存在 `outputs/benchmark/jetson_camera.csv`。报告中的当前四分类 Jetson 速度只能填写这个脚本在目标 Jetson 上生成的结果，不能用 Windows FPS 或同组五分类模型的数字替代。
