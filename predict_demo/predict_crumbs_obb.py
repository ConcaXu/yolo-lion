import numpy as np

from ultralytics import YOLO


def predict():
    model = YOLO("../model/best-obb-v2.pt")
    return model.predict(source='../source_demo/crumbs_3.jpg', save=True,
                         imgsz=640, conf=0.3, retina_masks=True, classes=[2])


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


if __name__ == '__main__':
    results = predict()
    res = obb_result_to_box(results)
    print("res", res)
    # for result in results:
    #     data_numpy = result.obb.data.cpu().numpy()
    #     for data in data_numpy:
    #         print("data", data)
