import cv2
import numpy as np

# 全局参数默认值
params = {
    'clahe_clip': 2.0,
    'clahe_grid': 8,
    'sobel_ksize': 3,
    'thresh': 50,
    'morph_h': 15,
    'hough_thresh': 50,
    'min_length': 30,
    'max_gap': 10
}

def update_image(*args):
    # 重新处理并显示图像
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 使用当前参数值处理图像
    clahe = cv2.createCLAHE(
        clipLimit=params['clahe_clip'],
        tileGridSize=(params['clahe_grid'], params['clahe_grid'])
    )
    enhanced = clahe.apply(gray)
    
    sobel_y = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=params['sobel_ksize']|1)
    edges = cv2.convertScaleAbs(sobel_y)
    
    _, binary = cv2.threshold(edges, params['thresh'], 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, params['morph_h']))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # 5. 可选：过滤非垂直线条
    lines = cv2.HoughLinesP(opened, 1, np.pi / 180, threshold=params['hough_thresh'], minLineLength=params['min_length'], maxLineGap=params['max_gap'])
    result = np.zeros_like(opened)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            if 80 < abs(angle) < 100:  # 保留接近垂直的线
                cv2.line(result, (x1, y1), (x2, y2), 255, 2)

    # 拼接所有处理阶段的图像
    # 统一所有图像为单通道
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR) if len(enhanced.shape) == 2 else enhanced
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR) if len(edges.shape) == 2 else edges
    binary = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR) if len(binary.shape) == 2 else binary
    opened = cv2.cvtColor(opened, cv2.COLOR_GRAY2BGR) if len(opened.shape) == 2 else opened
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR) if len(result.shape) == 2 else result

    # 统一图像尺寸
    h, w = enhanced.shape[:2]
    target_size = (w, h)
    
    edges = cv2.resize(edges, target_size)
    binary = cv2.resize(binary, target_size)
    opened = cv2.resize(opened, target_size)
    result = cv2.resize(result, target_size)

    # 创建两行布局
    row1 = cv2.hconcat([enhanced, edges, binary])
    row2 = cv2.hconcat([opened, result, np.zeros_like(enhanced)])  # 添加空白占位
    
    combined = cv2.vconcat([row1, row2])
    
    # 调整最终显示尺寸
    combined = cv2.resize(combined, (0,0), fx=0.8, fy=0.8)
    cv2.imshow('Processing Pipeline', combined)

# ===== 主程序 =====
if __name__ == "__main__":
    image_path = "../source_demo/crumbs_yuan.jpg"
    
    # 创建控制窗口
    cv2.namedWindow('Processing Pipeline', cv2.WINDOW_NORMAL)
    
    # 创建轨迹栏
    cv2.createTrackbar('CLAHE Clip', 'Processing Pipeline', 2, 10, 
                      lambda v: params.update({'clahe_clip': v/10.0}) or update_image())
    cv2.createTrackbar('CLAHE Grid', 'Processing Pipeline', 8, 20, 
                      lambda v: params.update({'clahe_grid': v}) or update_image())
    cv2.createTrackbar('Sobel KSize', 'Processing Pipeline', 3, 7, 
                      lambda v: params.update({'sobel_ksize': v}) or update_image())
    cv2.createTrackbar('Threshold', 'Processing Pipeline', 50, 255, 
                      lambda v: params.update({'thresh': v}) or update_image())
    cv2.createTrackbar('Morph H', 'Processing Pipeline', 15, 50, 
                      lambda v: params.update({'morph_h': v}) or update_image())
    
    # 初始化显示
    update_image()
    cv2.waitKey(0)
    cv2.destroyAllWindows()