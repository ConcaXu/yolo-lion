import yaml
from PIL import Image, ImageDraw, ImageFont
import os
from datetime import datetime
import logging

from ultralytics import YOLO

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S.%f'[:-3]
)

# 提前加载YAML配置文件
with open("D:/PythonProject/bread_inspect/model/breads950.yaml", 'r', encoding='utf-8') as file:
    YAML_CONTENT = yaml.safe_load(file)


def draw_boxes_and_save(image, image_path, boxes):
    """
    根据检测结果绘制边界框并保存图片（优化IO效率版本）

    Args:
        image: PIL.Image对象
        image_path: 原始图片路径（用于记录）
        boxes: 检测结果的边界框数据
    """
    try:
        start_time = datetime.now()

        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype('arial.ttf', 20)

        for data in boxes:
            x_min, y_min, x_max, y_max, confidence, class_id = data
            class_name = YAML_CONTENT['names'][int(class_id)]

            # 绘制边界框
            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)

            # 绘制标签
            text = f"{class_name}: {confidence:.2f}"
            draw.text((x_min, max(y_min - 10, 0)), text, fill="white", font=font)
            logging.debug(f"Drawn box: {text}")

        # 保存路径处理
        ng_path = "../ng_path"
        os.makedirs(ng_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        save_path = os.path.join(ng_path, f"{timestamp}.jpg")

        image.save(save_path)
        time_cost = (datetime.now() - start_time).total_seconds() * 1000

        logging.info(f"Image saved to {save_path} | Drawing time: {time_cost:.3f}ms")
        return True

    except Exception as e:
        logging.error(f"Error in drawing boxes: {str(e)}", exc_info=True)
        return False


def load_model(model_path):
    """模型加载函数（带耗时统计）"""
    try:
        logging.info("Starting model loading...")
        start_time = datetime.now()

        model = YOLO(model_path)

        load_time = (datetime.now() - start_time).total_seconds() * 1000
        logging.info(f"Model loaded successfully | Load time: {load_time:.3f}ms")
        return model

    except Exception as e:
        logging.error(f"Model loading failed: {str(e)}", exc_info=True)
        raise


def predict(model, num):
    """主预测流程（包含完整耗时统计）"""
    try:
        # 模型加载

        # 执行预测
        logging.info("Starting prediction...")
        predict_start = datetime.now()

        results = model.predict("crumbs.jpg",
                                save=False,
                                imgsz=640,
                                conf=0.3,
                                verbose=False)  # 禁用内置输出

        predict_time = (datetime.now() - predict_start).total_seconds() * 1000
        logging.info(f"Prediction completed | Total predict time: {predict_time:.3f}ms,模型---{num}---")

        # 处理结果
        for idx, result in enumerate(results):
            try:
                # 结果解析
                data_numpy = result.boxes.data.cpu().numpy()
                orig_image = Image.fromarray(result.orig_img[..., ::-1])  # BGR转RGB

                logging.info(f"Processing result {idx + 1} | Boxes detected: {len(data_numpy)},模型---{num}---")

                # 绘图保存
                if len(data_numpy) > 0:
                    draw_success = draw_boxes_and_save(orig_image, result.path, data_numpy)
                    if not draw_success:
                        logging.warning(f"Failed to draw boxes for result {idx + 1}")
                else:
                    logging.info(f"No boxes detected in result {idx + 1}")

            except Exception as e:
                logging.error(f"Error processing result {idx + 1}: {str(e)}", exc_info=True)

        return True

    except Exception as e:
        logging.error(f"Critical error in prediction pipeline: {str(e)}", exc_info=True)
        return False


if __name__ == "__main__":
    logging.info("========== Start Detection Pipeline ==========")
    model1 = load_model("../model/best-M-C2f_par.pt")
    model2 = load_model("../model/best-obb.pt")
    for i in range(0, 20):
        pipeline_start = datetime.now()
        success1 = predict(model1, 1)
        # success2 = predict(model2, 2)
        total_time = (datetime.now() - pipeline_start).total_seconds() * 1000
        # status = "SUCCESS" if success1 and success2 else "FAILED"
        status = "SUCCESS" if success1 else "FAILED"
        logging.info(f"Pipeline {status} | Total pipeline time: {total_time:.3f}ms")
