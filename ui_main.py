import sys
import os
import cv2
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QComboBox, QTextEdit
)
from yolo_infer import YOLOInfer  # 确保yolo_infer.py与本文件同目录

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('YOLO11 眼动跟踪应用')
        self.setGeometry(100, 100, 650, 430)
        self.inferer = None

        # 动态获取 models 文件夹下的所有 .pt 模型文件
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_folder = os.path.join(base_dir, "models")
        self.model_paths = [os.path.join(model_folder, f) for f in os.listdir(model_folder) if f.endswith('.pt')]
        self.selected_video_path = None
        self.selected_image_path = None
        self.init_ui()

    def init_ui(self):
        # 模型选择
        self.model_combo = QComboBox()
        self.model_combo.addItems([os.path.basename(p) for p in self.model_paths])
        self.btn_load_model = QPushButton('加载模型')
        self.btn_load_model.clicked.connect(self.load_model)

        # 去噪方式选择
        self.denoise_combo = QComboBox()
        self.denoise_combo.addItems(['无', '高斯滤波', '中值滤波', '均值滤波'])
        self.denoise_combo.setCurrentIndex(0)

        self.btn_select_video = QPushButton('选择视频')
        self.btn_select_video.clicked.connect(self.select_video)
        self.btn_select_image = QPushButton('选择图片')
        self.btn_select_image.clicked.connect(self.select_image)
        self.btn_track = QPushButton('视频检测')
        self.btn_track.clicked.connect(self.track_video)
        self.btn_pic_detect = QPushButton('图片检测')
        self.btn_pic_detect.clicked.connect(self.detect_image)

        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)

        top_layout = QHBoxLayout()
        top_layout.addWidget(QLabel('选择模型:'))
        top_layout.addWidget(self.model_combo)
        top_layout.addWidget(self.btn_load_model)
        top_layout.addSpacing(20)
        top_layout.addWidget(QLabel('去噪方式:'))
        top_layout.addWidget(self.denoise_combo)

        btn_layout = QVBoxLayout()
        btn_layout.addWidget(self.btn_select_video)
        btn_layout.addWidget(self.btn_select_image)
        btn_layout.addWidget(self.btn_pic_detect)
        btn_layout.addWidget(self.btn_track)

        main_layout = QHBoxLayout()
        main_layout.addLayout(btn_layout)
        main_layout.addWidget(self.log_edit)

        layout = QVBoxLayout()
        layout.addLayout(top_layout)
        layout.addLayout(main_layout)
        self.setLayout(layout)

    def get_denoise_method(self):
        idx = self.denoise_combo.currentIndex()
        # 和yolo_infer.py的denoise_method参数对应
        if idx == 1:
            return "gaussian"
        elif idx == 2:
            return "median"
        elif idx == 3:
            return "mean"
        else:
            return None

    def load_model(self):
        idx = self.model_combo.currentIndex()
        model_path = self.model_paths[idx]
        self.inferer = YOLOInfer(model_path)
        self.log_edit.append(f"加载模型: {os.path.basename(model_path)}")

    def select_video(self):
        path, _ = QFileDialog.getOpenFileName(self, '选择视频文件', '', 'Video Files (*.mp4 *.avi)')
        if path:
            self.selected_video_path = path
            self.log_edit.append(f"已选择视频: {path}")

    def select_image(self):
        path, _ = QFileDialog.getOpenFileName(self, '选择图片文件', '', 'Image Files (*.jpg *.png *.bmp)')
        if path:
            self.selected_image_path = path
            self.log_edit.append(f"已选择图片: {path}")

    def detect_image(self):
        if not self.inferer or not self.selected_image_path:
            self.log_edit.append("请先加载模型并选中图片文件！")
            return
        try:
            denoise_method = self.get_denoise_method()
            results, annotated = self.inferer.infer_image(self.selected_image_path, denoise_method=denoise_method)
            cv2.imshow("检测结果", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            self.log_edit.append(f"图片检测完成。去噪方式: {self.denoise_combo.currentText()}")
        except Exception as e:
            import traceback
            self.log_edit.append(f"图片检测出错: {e}")
            print(traceback.format_exc())

    def track_video(self):
        if not self.inferer or not self.selected_video_path:
            self.log_edit.append("请先加载模型并选择视频！")
            return
        try:
            denoise_method = self.get_denoise_method()
            self.showMinimized()  # 主窗口最小化

            window_name = "跟踪窗口 - 按Q退出"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 960, 720)

            def on_result(frame, annotated, results):
                cv2.imshow(window_name, annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    raise StopIteration()

            self.inferer.infer_video(self.selected_video_path, on_result, denoise_method=denoise_method)
        except StopIteration:
            pass
        finally:
            cv2.destroyAllWindows()
            self.showNormal()  # 恢复主窗口
            self.log_edit.append("视频检测完成。")  # ≤50字

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())