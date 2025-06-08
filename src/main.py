import os
import json
import cv2
from process import process_img

# ===== 路径设置 =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.abspath(os.path.join(BASE_DIR, '.', 'images'))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
OUTPUT_JSON = os.path.join(BASE_DIR, 'output_results.json')
OUTPUT_IMG_DIR = os.path.join(BASE_DIR, 'output', 'images')  


os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)  
# ===== 推理并保存结果 =====
all_results = {}

for fname in os.listdir(IMAGE_DIR):
    if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    img_path = os.path.join(IMAGE_DIR, fname)
    try:
        result = process_img(img_path)
        all_results[fname] = result

        # # ✅ 读取原图并绘制检测框
        # img = cv2.imread(img_path)
        # for box in result:
        #     x, y, w, h = box["x"], box["y"], box["w"], box["h"]
        #     pt1, pt2 = (x, y), (x + w, y + h)
        #     cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)
        #     cv2.putText(img, "det", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # # ✅ 保存绘制后的图像
        # save_path = os.path.join(OUTPUT_IMG_DIR, fname)
        # cv2.imwrite(save_path, img)

        print(f"✅ 处理完成: {fname}, 目标数: {len(result)}")
    except Exception as e:
        print(f"❌ 错误处理 {fname}: {e}")
        all_results[fname] = []

# ===== 保存为 JSON 文件 =====
with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print(f"📄 所有结果已保存到: {OUTPUT_JSON}")
print(f"🖼 检测框图片已保存至: {OUTPUT_IMG_DIR}")