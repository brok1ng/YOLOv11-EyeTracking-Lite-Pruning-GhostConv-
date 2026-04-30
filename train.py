from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'C:\Desktop\graduate_design\eyetrack_app\yolov11_prune_main\ultralytics\cfg\models\11\yolo11.yaml')#改为自己的yolo11配置文件
    model.train(data=r"C:\Desktop\graduate_design\eyetrack_app\yolov11_prune_main\eye.yaml",
                imgsz=640,
                epochs=100,
                batch=16,
                project='runs_ghost',  # 项目文件夹的名，默认为runs
                name='exp',  # 用于保存训练文件夹名，默认exp，依次累加
                device=0,  # 要运行的设备 device =0 是GPU显卡训练，device = cpu
                lr0=0.001,
                lrf=1e-5,
                cos_lr=True 
                )#data该为自己的数据集配置文件路径。