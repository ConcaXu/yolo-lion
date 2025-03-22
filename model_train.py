from ultralytics import YOLO

def train():
    model = YOLO('yolov8n-p2.yaml').load( 'yolov8n.pt')
    # results = model.train(data='datasets/BreadAllTrainVal2500/BreadAllTrainVal2500.yaml', D:\PythonProject\mayercnc-bakeware-python\runs\detect\train\weights\last.pt, 7.7MB
    results = model.train(data='datasets/BreadAllTrainVal2500/BreadAllTrainVal2500.yaml',
                          epochs=300, batch=8,
                          device=0, workers=16,
                          imgsz=640,resume=False,
                          amp=False,lr0=0.01,lrf=0.01,
                          plots=True,perspective=0.0005,
                          degrees=10)
    return results

if __name__ == '__main__':
    results = train()
    # print(results)
