#  自动捡网球机器人

本项目基于 YOLOv8 ，实现了图像中网球的自动检测，并输出最终目标坐标，支持批量图像处理、结果保存及性能评估

---

##  环境部署说明

###  Python 版本

推荐使用：
```
Python 3.10.4
```

建议使用虚拟环境（ conda）进行依赖管理。

### 安装依赖

请运行以下命令安装项目所需依赖：

```bash
pip install -r requirements.txt
```

---

##  快速开始

### 执行主程序：

```bash
python main.py
```

程序将自动读取图像、执行检测，并保存结果。

---

##  项目结构说明

```bash
project-root/
├── main.py               # 主程序入口，执行推理与保存
├── process.py            # 模型推理逻辑
├── models/
│   └── best.onnx         # 训练好的 YOLOv8 ONNX 模型
├── src/
│   └── images/           #  测试图像请放在此目录
├── output/
│   ├── output_results.json  # 所有检测结果 JSON 输出
│   └── images/              # 带检测框的图像输出（如果想要图像输出可将代码图像输出代码取消注释)
├── requirements.txt      # 项目依赖列表
└── README.md             # 本说明文档
```

---

##  注意事项

- 所有测试图像应放置于 **`src/images/`** 目录；
- 模型文件需命名为 `best.onnx` 并放置在 `models/` 文件夹下；
- 程序将自动创建 `output/` 文件夹并输出检测结果；（未注释的话）
- 输出的 JSON 包含所有图像对应的目标框坐标信息。

---

