# from ultralytics import YOLO

# # Load a model
# # model = YOLO("yolov8n.yaml")  # build a new model from scratch
# model = YOLO("yolov8n-seg.yaml")  # build a new model from YAML
# model = YOLO("yolov8n-seg.pt")  # load a pretrained model (recommended for training)
# model = YOLO("yolov8n-seg.yaml").load("yolov8n.pt")  # build from YAML and transfer weights
# # Use the model
# results = model.train(data="/home/duy/Downloads/train-yolov8-custom-dataset-step-by-step-guide-master/local_env/config.yaml", epochs=3000, batch=-1)  # train the model
# results = model.val()
# #/home/duy/.local/share/pipx/venvs/ultralytics/bin/python train.py








from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO("/workspace/SignDetect/weights/best.pt")

# Define path to the image file
source = "/workspace/img/img/imgg_1437.jpg"

# Run inference on the source
results = model(source, stream=True)

for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()
    result.save(filename="stop_yolov8.jpg")






# from ultralytics import YOLO

# # Load a pretrained YOLOv8n model
# model = YOLO("/home/duy/Downloads/train-yolov8-custom-dataset-step-by-step-guide-master/runs/detect/train11/weights/best.pt")

# # Run inference on an image
# results = model("/home/duy/Downloads/train-yolov8-custom-dataset-step-by-step-guide-master/local_env/data/images/train/img_303.jpg")  # results list

# # View results
# for r in results:
#     print(r.boxes)  # print the Boxes object containing the detection bounding boxes


