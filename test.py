from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    # model = YOLO(r'yolov8m-p2.yaml')  # build a new model from YAML
    model = YOLO(r'yolov8m-obb.yaml')  # build a new model from YAML
    model.info()
