from ultralytics import YOLO
# 加载预训练的 YOLOv11n 模型
model = YOLO('yolo11x.pt')
source = '哪吒.jpg' #更改为自己的图片路径
#model = YOLOv11("yolov11n.pt", device="cuda:0")
#model.predict(source="input.mp4", imgsz=1280)
model.predict(source, save=True,show=True,)

from ultralytics import YOLO

if __name__ == '__main__':
    # 加载模型

    model = YOLO("yolo11n.pt")
    # 进行推理
    model.predict(source="https://ultralytics.com/images/bus.jpg"
                  ,  # source是要推理的图片路径这里使用yolo自带的图片
                  save=True,  # 是否在推理结束后保存结果
                  show=True,  # 是否在推理结束后显示结果,  # 结果的保存路径
                  )
