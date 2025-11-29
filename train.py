from ultralytics import YOLO

def main():
    # Load pretrained model (relative path)
    model = YOLO("yolov8m.pt")

    # Data configuration (relative path)
    data_yaml = "ultralytics/cfg/datasets/foraminifera.yaml"

    # Training arguments
    train_args = dict(
        data=data_yaml,
        epochs=300,
        imgsz=1024,
        batch=8,
        device=[0, 1],
        workers=8,

        project="runs",
        name="detect",

        # Augmentations (optimized from experiments)
        mosaic=0.9,
        copy_paste=0.25,
        mixup=0.01,

        hsv_h=0.015,
        hsv_s=0.65,
        hsv_v=0.38,

        degrees=10.0,
        translate=0.10,
        scale=0.45,
        shear=2.0,
        perspective=0.0,

        fliplr=0.5,
        flipud=0.0,

        auto_augment="autoaugment",
    )

    print("Starting training...")
    results = model.train(**train_args)

    print("\nTraining finished.")
    print("Results saved in: runs/detect")
    print("Best weights: runs/detect/weights/best.pt")

if __name__ == "__main__":
    main()
