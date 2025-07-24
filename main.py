import sys
import json
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QTextEdit, QFileDialog, 
                             QLabel, QLineEdit, QFormLayout)
from PyQt5.QtGui import QPainter, QColor, QPen, QImage, QFont
from PyQt5.QtCore import Qt, QPoint, QThread, pyqtSignal
import Function
import Pool
import Softmax
import Conv33
import os

class DrawingCanvas(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        self.setFixedSize(280, 280)
        self.image = QImage(self.size(), QImage.Format_RGB32)
        self.image.fill(Qt.white)
        self.drawing = False
        self.lastPoint = QPoint()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(self.rect(), self.image, self.image.rect())
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.lastPoint = event.pos()
            
    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.LeftButton) and self.drawing:
            painter = QPainter(self.image)
            painter.setPen(QPen(Qt.black, 10, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawLine(self.lastPoint, event.pos())
            self.lastPoint = event.pos()
            self.update()
            
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False
            
    def get_image_array(self):
        image = self.image.scaled(28, 28, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        ptr = image.bits()
        ptr.setsize(image.byteCount())
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape(28, 28, 4)
        gray = arr[:, :, 0]
        normalized = 1.0 - gray / 255.0
        return normalized
    
    def clear_canvas(self):
        self.image.fill(Qt.white)
        self.update()

class TrainingThread(QThread):
    signal_update_text = pyqtSignal(str)

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window

    def run(self):
        try:
            epochs = int(self.main_window.epochs_input.text())
            dataset_size = int(self.main_window.dataset_input.text())
            learning_rate = float(self.main_window.lr_input.text())
            
            self.signal_update_text.emit(f"训练参数 - 轮数: {epochs}, 数据集大小: {dataset_size}, 学习率: {learning_rate}")
            
            train_images = Function.parse_mnist(minst_file_addr="your file addr")  # 训练集图像
            train_labels = Function.parse_mnist(minst_file_addr="your file addr")  # 训练集标签

            self.signal_update_text.emit('MNIST CNN initialized!')
            loss = 0
            num_correct = 0
            for epoch in range(epochs):
                self.signal_update_text.emit('--- Epoch %d ---' % (epoch + 1))
            
                # Shuffle the training data
                permutation = np.random.permutation(len(train_images))
                train_images = train_images[permutation]
                train_labels = train_labels[permutation]
            
                # Train!
                loss = 0
                num_correct = 0
            
                # i: index
                # im: image
                # label: label
                for i, (im, label) in enumerate(zip(train_images, train_labels)):
                    if i == dataset_size or i == 10000:
                        break

                    if i > 0 and i % 100 == 99:
                        message = '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' % (i + 1, loss / 100, num_correct)
                        self.signal_update_text.emit(message)
                        print(message)
                        loss = 0
                        num_correct = 0

                    l, acc = self.main_window.train(im, label)
                    loss += l
                    num_correct += acc
            self.signal_update_text.emit('Train finished!')
        except ValueError as e:
            self.signal_update_text.emit(f"参数错误: {e}")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

        self.conv    = Conv33.Conv33(8)
        self.pool    = Pool.Pool()
        self.softmax = Softmax.Softmax(13*13*8, 10)
        
    def initUI(self):
        self.setWindowTitle('手写数字识别')
        self.setGeometry(100, 100, 800, 600)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        left_layout = QVBoxLayout()
        
        self.canvas = DrawingCanvas()
        left_layout.addWidget(self.canvas)
        
        self.text_output = QTextEdit()
        self.text_output.setReadOnly(True)
        
        # 加大输出框字体
        output_font = QFont()
        output_font.setPointSize(12)
        self.text_output.setFont(output_font)
        
        left_layout.addWidget(self.text_output)
        
        right_layout = QVBoxLayout()
        
        # 加粗按钮字体
        button_font = QFont()
        button_font.setBold(True)
        button_font.setPointSize(12)
        
        self.btn_load = QPushButton('导入权重')
        self.btn_save = QPushButton('导出权重')
        self.btn_reset = QPushButton('重置绘画框')
        self.btn_recognize = QPushButton('识别绘画框')
        self.btn_train = QPushButton('训练')
        
        for btn in [self.btn_load, self.btn_save, self.btn_reset, 
                   self.btn_recognize, self.btn_train]:
            btn.setFont(button_font)
            btn.setMinimumHeight(60)
            right_layout.addWidget(btn)
        
        # 使用QFormLayout确保输入框对齐
        param_layout = QFormLayout()
        param_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        
        # 加大输入框字体
        input_font = QFont()
        input_font.setPointSize(12)
        
        # 加大标签字体
        label_font = QFont()
        label_font.setPointSize(12)
        
        # 训练轮数
        epochs_label = QLabel("训练轮数:")
        epochs_label.setFont(label_font)
        self.epochs_input = QLineEdit("10")
        self.epochs_input.setFont(input_font)
        param_layout.addRow(epochs_label, self.epochs_input)
        
        # 训练数据集数量
        dataset_label = QLabel("训练数据集数量:")
        dataset_label.setFont(label_font)
        self.dataset_input = QLineEdit("100")
        self.dataset_input.setFont(input_font)
        param_layout.addRow(dataset_label, self.dataset_input)
        
        # 学习率
        lr_label = QLabel("学习率:")
        lr_label.setFont(label_font)
        self.lr_input = QLineEdit("0.005")
        self.lr_input.setFont(input_font)
        param_layout.addRow(lr_label, self.lr_input)
        
        # 设置固定宽度确保所有输入框等宽
        input_width = 150
        self.epochs_input.setFixedWidth(input_width)
        self.dataset_input.setFixedWidth(input_width)
        self.lr_input.setFixedWidth(input_width)
        
        # 将参数布局添加到右侧布局
        param_widget = QWidget()
        param_widget.setLayout(param_layout)
        right_layout.addWidget(param_widget)
        
        # 在右侧布局底部添加拉伸，使按钮居上
        right_layout.addStretch(1)
        
        main_layout.addLayout(left_layout, 7)
        main_layout.addLayout(right_layout, 3)
        
        self.weights = {
            'filters': [],
            'biases': [],
            'weights': [],
        }
        
        self.btn_load.clicked.connect(self.load_weights)
        self.btn_save.clicked.connect(self.save_weights)
        self.btn_reset.clicked.connect(self.reset_canvas)
        self.btn_recognize.clicked.connect(self.recognize_digit)
        self.btn_train.clicked.connect(self.train_network)
        
        self.show()
        
    def load_weights(self):
        filename, _ = QFileDialog.getOpenFileName(self, '导入权重', '', 'JSON Files (*.json)')
        if filename:
            try:
                with open(filename, 'r') as f:
                    self.weights = json.load(f)
                    self.conv.filters    = np.array(self.weights['filters'])
                    self.softmax.weights = np.array(self.weights['weights'])
                    self.softmax.biases  = np.array(self.weights['biases'])
                self.text_output.append(f"成功从 {filename} 导入权重")
            except Exception as e:
                self.text_output.append(f"导入权重失败: {e}")
    
    def save_weights(self):
        filename, _ = QFileDialog.getSaveFileName(self, '导出权重', '', 'JSON Files (*.json)')
        
        self.weights['filters'] = self.conv.filters.tolist()
        self.weights['biases']  = self.softmax.biases.tolist()
        self.weights['weights'] = self.softmax.weights.tolist()

        if filename:
            try:
                with open(filename, 'w') as f:
                    json.dump(self.weights, f)
                    self.text_output.append(f"成功将权重保存到 {filename}")
            except Exception as e:
                self.text_output.append(f"保存权重失败: {e}")
    
    def reset_canvas(self):
        self.canvas.clear_canvas()
        self.text_output.append("已重置绘画框")
    
    def recognize_digit(self):
        image_array = self.canvas.get_image_array()
        
        image_array = (image_array * 255).astype(np.uint8)
        #print(image_array)
        # label随便输入就好了 label是用来验证acc的，我们不关心，只需要预测结果
        out, _, _ = self.forward(image_array, 0)
        print(out)
        label = np.argmax(out)
        probability = np.max(out)
        self.text_output.append(f"识别结果: {label}, 概率为: {probability}")
    
    def forward(self, image, label):
        out = self.conv.forward((image / 255) - 0.5)
        out = self.pool.forward(out)
        out = self.softmax.forward(out)

        loss = -np.log(out[label])
        
        acc = 1 if np.argmax(out) == label else 0

        return out, loss, acc

    def train(self, im, label, lr=.005):
        # Forward
        out, loss, acc = self.forward(im, label)

        # 初始化梯度
        gradient = np.zeros(10)
        gradient[label] = -1 / out[label]
    
        # 反向传播
        gradient = self.softmax.backprop(gradient, lr)
        gradient = self.pool.backprop(gradient)
        gradient = self.conv.backprop(gradient, lr)
    
        return loss, acc
    
    def train_network(self):
        self.training_thread = TrainingThread(self)
        self.training_thread.signal_update_text.connect(self.update_text_output)
        self.training_thread.start()

    def update_text_output(self, message):
        self.text_output.append(message)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())