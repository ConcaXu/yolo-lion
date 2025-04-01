from ultralytics import YOLO


def train():
    model = YOLO('yolov8m-obb.yaml').load('yolov8m-obb.pt')
    results = model.train(
        data='datasets/only_hair_train/only_hair_train.yaml',
        epochs=500, batch=8,
        device=0, workers=8,
        imgsz=640, resume=False,
        amp=False, lr0=0.01, lrf=0.01,
        plots=True, perspective=0.0005,
        degrees=10)
    return results


if __name__ == '__main__':
    results = train()
    print(results)
