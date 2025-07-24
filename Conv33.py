import numpy as np

class Conv33:
    def __init__(self, filter_num):
        # filter数量
        self.filter_num = filter_num
        # 随机生成filters规格为 3*3
        self.filters = np.random.randn(filter_num, 3, 3) / 9

    def forward(self, img):
        height, width = img.shape
        self.img    = img
        self.height = height
        self.width  = width

        res = np.zeros((height-2, width-2, self.filter_num))

        for i in range(height - 2):
            for j in range(width - 2):
                # 提取3*3图像块
                img_block = img[i:(i+3), j:(j+3)]
                # 3*3块求和
                res[i, j] = np.sum(img_block * self.filters, axis=(1,2))
        
        return res

    def backprop(self, d_L_d_out, learn_rate):
        # 初始化梯度
        d_L_d_filters = np.zeros(self.filters.shape)
        
        for i in range(self.height - 2):
            for j in range(self.width - 2):
                # 提取3*3图像块
                img_block = self.img[i:(i+3), j:(j+3)]

                for f in range(self.filter_num):
                    d_L_d_filters[f] += d_L_d_out[i, j, f] * img_block

        self.filters -= learn_rate * d_L_d_filters

        return None
