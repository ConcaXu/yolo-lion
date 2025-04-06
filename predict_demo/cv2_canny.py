import cv2
import numpy as np
import matplotlib.pyplot as plt

from ultralytics import YOLO


def predict(img_path):
    model = YOLO("../model/best-obb-v2.pt")
    return model.predict(source=img_path, save=True,
                         imgsz=640, conf=0.3, retina_masks=True)


def obb_to_aabb(xc, yc, w, h, theta, confidence, class_id):
    """
    将旋转矩形 (OBB) 转换为最小外接矩形 (AABB)
    :param xc: 旋转矩形中心 x 坐标
    :param yc: 旋转矩形中心 y 坐标
    :param w: 旋转矩形的宽度
    :param h: 旋转矩形的高度
    :param theta: 旋转角度（弧度）
    :return: (xmin, ymin, xmax, ymax) -> AABB 边界
    """
    # 计算 OBB 的四个角点（相对中心点）
    corners = np.array([
        [-w / 2, -h / 2],
        [w / 2, -h / 2],
        [w / 2, h / 2],
        [-w / 2, h / 2]
    ])

    # 旋转矩阵
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    # 旋转角点坐标
    rotated_corners = np.dot(corners, rotation_matrix.T) + np.array([xc, yc])

    # 计算 AABB
    xmin, ymin = np.min(rotated_corners, axis=0)
    xmax, ymax = np.max(rotated_corners, axis=0)

    return xmin, ymin, xmax, ymax, confidence, class_id


def obb_result_to_box(results):
    res = []
    for result in results:
        if result.obb.data is None:
            continue
        else:
            data_numpy = result.obb.data.cpu().numpy()
            temp_numpy = []
            for data in data_numpy:
                obb_box = data[0], data[1], data[2], data[3], data[4], data[5], data[6]  # 30 度旋转
                aabb = obb_to_aabb(*obb_box)
                temp_numpy.append(aabb)
            res.append(np.array(temp_numpy))
    return res


def edge_detection_demo(image_path):
    """图片边缘检测（仅保留Scharr方法）"""
    # 读取图片并转为灰度图
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"图片未找到: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # 准备可视化画布
    plt.figure(figsize=(12, 5))

    # 原始图像
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')

    # Scharr边缘检测
    scharr_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
    scharr = cv2.addWeighted(cv2.convertScaleAbs(scharr_x), 0.5,
                             cv2.convertScaleAbs(scharr_y), 0.5, 0)

    plt.subplot(1, 2, 2)
    plt.imshow(scharr, cmap='gray')
    plt.title("Scharr Edge")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return {
        "original": img,
        "scharr": scharr
    }


def dynamic_scharr_demo(image_path):
    """动态Scharr边缘检测调试工具"""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"图片未找到: {image_path}")

    # 初始化参数
    params = {
        'blur_size': 5,
        'scale': 0.5,
        'delta': 0,
        'threshold': 50
    }

    # 创建窗口
    cv2.namedWindow('Scharr Demo', cv2.WINDOW_NORMAL)

    # 创建滑动条
    cv2.createTrackbar('模糊核大小', 'Scharr Demo', 5, 15, lambda x: None)
    cv2.createTrackbar('缩放比例', 'Scharr Demo', 50, 100, lambda x: None)
    cv2.createTrackbar('亮度阈值', 'Scharr Demo', 50, 255, lambda x: None)

    while True:
        # 获取当前滑动条值
        params['blur_size'] = cv2.getTrackbarPos('模糊核大小', 'Scharr Demo') | 1  # 保证奇数
        params['scale'] = cv2.getTrackbarPos('缩放比例', 'Scharr Demo') / 100.0
        threshold = cv2.getTrackbarPos('亮度阈值', 'Scharr Demo')

        # 预处理
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (params['blur_size'], params['blur_size']), 0)

        # Scharr边缘检测
        scharr_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
        scharr_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)

        # 合并并增强显示
        scharr = cv2.addWeighted(cv2.convertScaleAbs(scharr_x), params['scale'],
                                 cv2.convertScaleAbs(scharr_y), params['scale'], 0)

        # 应用阈值
        _, scharr_thresh = cv2.threshold(scharr, threshold, 255, cv2.THRESH_BINARY)

        # 显示结果
        combined = np.hstack([gray, scharr, scharr_thresh])
        cv2.imshow('Scharr Demo', combined)

        # 退出条件
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC退出
            break

    cv2.destroyAllWindows()
    return scharr_thresh


def optimized_scharr(image_path, blur_size=11, scale=0.65, threshold=90):
    """固定参数Scharr边缘检测"""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"图片未找到: {image_path}")

    # 预处理（自动保证模糊核为奇数）
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (blur_size | 1, blur_size | 1), 0)

    # Scharr边缘检测
    scharr_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
    scharr = cv2.addWeighted(cv2.convertScaleAbs(scharr_x), scale,
                             cv2.convertScaleAbs(scharr_y), scale, 0)

    # 应用阈值
    _, scharr_thresh = cv2.threshold(scharr, threshold, 255, cv2.THRESH_BINARY)

    # 可视化对比
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(scharr, cmap='gray')
    plt.title(f"Scharr (scale={scale})")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(scharr_thresh, cmap='gray')
    plt.title(f"Threshold={threshold}")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return scharr_thresh


# ===== 使用示例 =====
if __name__ == "__main__":
    img_path = "../source_demo/crumbs_yuan.jpg"
    # 使用您找到的最佳参数：blur_size=10会自动转为11（保证奇数）
    result = optimized_scharr(img_path,
                              blur_size=10,  # 实际会转为11
                              scale=0.65,  # 65/100
                              threshold=90)

    res = obb_result_to_box(predict(img_path))

    print(res)
    cv2.imwrite("optimized_scharr_final.jpg", result)
