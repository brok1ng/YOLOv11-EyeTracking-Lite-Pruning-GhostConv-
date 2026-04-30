import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QComboBox, QTextEdit
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

class YOLOApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('YOLO11 眼动跟踪应用')
        self.setGeometry(100, 100, 800, 600)
        self.init_ui()

    def init_ui(self):
        # 模型选择
        self.model_combo = QComboBox()
        self.model_combo.addItems(['模型1', '模型2'])  # 实际应从models目录读取
        self.btn_load_model = QPushButton('加载模型')
        self.btn_load_model.clicked.connect(self.load_model)

        # 图片/视频显示区
        self.display_label = QLabel('预览区')
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setFixedSize(400, 300)

        # 操作按钮
        self.btn_select_video = QPushButton('选择视频')
        self.btn_select_video.clicked.connect(self.select_video)
        self.btn_select_image = QPushButton('选择图片')
        self.btn_select_image.clicked.connect(self.select_image)
        self.btn_track = QPushButton('开始跟踪')
        self.btn_track.clicked.connect(self.track)
        self.btn_denoise = QPushButton('图片去噪')
        self.btn_denoise.clicked.connect(self.denoise)

        # 日志输出
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)

        # 布局
        top_layout = QHBoxLayout()
        top_layout.addWidget(QLabel('选择模型:'))
        top_layout.addWidget(self.model_combo)
        top_layout.addWidget(self.btn_load_model)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.display_label)
        left_layout.addWidget(self.btn_select_video)
        left_layout.addWidget(self.btn_select_image)
        left_layout.addWidget(self.btn_track)
        left_layout.addWidget(self.btn_denoise)

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout)
        main_layout.addWidget(self.log_edit)

        layout = QVBoxLayout()
        layout.addLayout(top_layout)
        layout.addLayout(main_layout)

        self.setLayout(layout)

    def load_model(self):
        model_name = self.model_combo.currentText()
        self.log_edit.append(f"加载模型: {model_name}")
        # TODO: 加载模型逻辑
        pass

    def select_video(self):
        path, _ = QFileDialog.getOpenFileName(self, '选择视频文件', '', 'Video Files (*.mp4 *.avi)')
        if path:
            self.log_edit.append(f"已选择视频: {path}")
            # TODO: 视频加载和预览
            pass

    def select_image(self):
        path, _ = QFileDialog.getOpenFileName(self, '选择图片文件', '', 'Image Files (*.jpg *.png *.bmp)')
        if path:
            self.log_edit.append(f"已选择图片: {path}")
            pixmap = QPixmap(path).scaled(self.display_label.size(), Qt.KeepAspectRatio)
            self.display_label.setPixmap(pixmap)
            # TODO: 图片加载
            pass

    def track(self):
        self.log_edit.append("执行跟踪操作...")
        # TODO: 跟踪逻辑
        pass

    def denoise(self):
        self.log_edit.append("执行去噪操作...")
        # TODO: 去噪逻辑
        pass

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = YOLOApp()
    win.show()
    sys.exit(app.exec_())