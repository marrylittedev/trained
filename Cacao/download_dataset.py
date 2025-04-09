from roboflow import Roboflow

rf = Roboflow(api_key="f4UBb9Y1BqAaVoiasTC1")
project = rf.workspace("cacaotrain").project("varieties")
version = project.version(5)
dataset = version.download("yolov5")
