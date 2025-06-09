import os
import cv2
import numpy as np
import onnxruntime as ort
import time


# ========== 模型加载设置 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best.onnx')
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name
img_size = 640

#
#参数:
#   img_path: 要识别的图片的路径
#
#返回:
#   返回结果为各赛题中要求的识别结果，具体格式可参考提供压缩包中的 “图片对应输出结果.txt” 中一张图片对应的结果
#
import os
import cv2
import numpy as np
import onnxruntime as ort

# 初始化模型
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best.onnx')
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
img_size = 640

def process_img(img_path):
    orig_img = cv2.imread(img_path)
    if orig_img is None:
        print(f"❌ 无法读取图像: {img_path}")
        return []

    H, W = orig_img.shape[:2]
    scale = min(img_size / H, img_size / W)
    new_h, new_w = int(H * scale), int(W * scale)
    resized = cv2.resize(orig_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_top = (img_size - new_h) // 2
    pad_left = (img_size - new_w) // 2
    input_img = np.full((img_size, img_size, 3), 114, dtype=np.uint8)
    input_img[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

    blob = cv2.dnn.blobFromImage(input_img, scalefactor=1 / 255.0, size=(img_size, img_size), swapRB=True, crop=False)
    outputs = session.run(None, {input_name: blob})
    pred = outputs[0].squeeze().T  # [C, N] -> [N, C]

    if pred.ndim != 2 or pred.shape[0] == 0:
        return []

    scores = pred[:, 4:]
    class_ids = np.argmax(scores, axis=1)
    conf_scores = scores[np.arange(len(scores)), class_ids]
    mask = conf_scores > 0.25

    pred = pred[mask]
    conf_scores = conf_scores[mask]
    class_ids = class_ids[mask]

    boxes = []
    raw_boxes = []

    for p in pred:
        cx, cy, w, h = p[:4]
        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h
        x1 = max(0, (x1 - pad_left) / scale)
        y1 = max(0, (y1 - pad_top) / scale)
        x2 = min(W, (x2 - pad_left) / scale)
        y2 = min(H, (y2 - pad_top) / scale)
        raw_boxes.append([x1, y1, x2, y2])

    raw_boxes = np.array(raw_boxes, dtype=np.float32)
    indices = cv2.dnn.NMSBoxes(raw_boxes.tolist(), conf_scores.tolist(), 0.25, 0.45)

    if indices is not None and len(indices) > 0:
        for idx in indices:
            i = int(idx) if np.isscalar(idx) else int(idx[0])
            x1, y1, x2, y2 = raw_boxes[i]
            box = {
                "x": int(x1),
                "y": int(y1),
                "w": int(x2 - x1),
                "h": int(y2 - y1)
            }
            boxes.append(box)

    return boxes



#
#以下代码仅作为选手测试代码时使用，仅供参考，可以随意修改
#但是最终提交代码后，process.py文件是作为模块进行调用，而非作为主程序运行
#因此提交时请根据情况删除不必要的额外代码
#
if __name__=='__main__':
    # imgs_folder = './images/'
    print("保存路径为：", os.getcwd())
    imgs_folder = os.path.join(os.path.dirname(__file__), 'images')
    img_paths = os.listdir(imgs_folder)
    def now():
        return int(time.time()*1000)
    last_time = 0
    count_time = 0
    max_time = 0
    min_time = now()
    for img_path in img_paths:
        print(img_path,':')
        last_time = now()
        # result = process_img(imgs_folder+img_path)
        result = process_img(os.path.join(imgs_folder, img_path))
        
        run_time = now() - last_time
        print('result:\n',result)
        print('run time: ', run_time, 'ms')
        print()
        count_time += run_time
        if run_time > max_time:
            max_time = run_time
        if run_time < min_time:
            min_time = run_time
    print('\n')
    print('avg time: ',int(count_time/len(img_paths)),'ms')
    print('max time: ',max_time,'ms')
    print('min time: ',min_time,'ms')
    # ✅ 保存检测框结果
    all_outputs = {}
    for img_path in img_paths:
        img_full_path = os.path.join(imgs_folder, img_path)
        result = process_img(img_full_path)
        all_outputs[img_path] = result

    import json
    with open('results.json', 'w', encoding='utf-8') as f:
        json.dump(all_outputs, f, ensure_ascii=False, indent=2)
    print('✅ 检测结果已保存至 results.json')

    # ✅ 保存性能统计结果
    with open('timing.txt', 'w', encoding='utf-8') as f:
        f.write(f'avg time: {int(count_time / len(img_paths))} ms\n')
        f.write(f'max time: {max_time} ms\n')
        f.write(f'min time: {min_time} ms\n')
    print('✅ 性能统计结果已保存至 timing.txt')
