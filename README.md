使用前需要下载PyQt5，使用命令

```shell
pip install PyQt5
```

或

```shell
pip3 install PyQt5 
```

同时还需要更改main.py中TrainingThread里mnist数据集的地址

```python
def run(self):
        try:
            epochs = int(self.main_window.epochs_input.text())
            dataset_size = int(self.main_window.dataset_input.text())
            learning_rate = float(self.main_window.lr_input.text())
            
            self.signal_update_text.emit(f"训练参数 - 轮数: {epochs}, 数据集大小: {dataset_size}, 学习率: {learning_rate}")
            
            train_images = Function.parse_mnist(minst_file_addr="your file addr")  # 训练集图像
            train_labels = Function.parse_mnist(minst_file_addr="your file addr")  # 训练集标签

```

点击运行后如下图所示

![pyqt运行截图](source\image.png)

右侧可以导入权重（提供了训练轮数3，数量10000，学习率0.005的权重文件），可以重置和识别绘画框，支持更改训练轮数，数据集数量，学习率，左侧上方可以绘制数字进行识别，下方则为提示信息