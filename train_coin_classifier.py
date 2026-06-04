# train_coin_classifier.py
# 한국 동전(10/50/100/500원) 분류기 학습.
# images/korean_coin 폴더의 ImageFolder 구조를 그대로 사용한다.

from ultralytics import YOLO

DATA_DIR = "images/korean_coin"
RUN_NAME = "coin_classifier"


def main():
    model = YOLO("yolov8n-cls.pt")
    model.train(
        data=DATA_DIR,
        epochs=60,
        imgsz=224,
        batch=32,
        device="cpu",
        project="runs/classify",
        name=RUN_NAME,
        exist_ok=True,
        patience=20,
        verbose=True,
    )


if __name__ == "__main__":
    main()
