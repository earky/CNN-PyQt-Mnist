import numpy as np

class Pool:
    def forward(self, conv):
        # 保存信息
        self.input = conv

        height, width, filter_num = conv.shape
        pool_height = height // 2
        pool_width  = width // 2

        self.pool_height = pool_height
        self.pool_width  = pool_width

        res = np.zeros((pool_height, pool_width, filter_num))

        for i in range(pool_height):
            for j in range(pool_width):
                # 获取2*2conv块
                conv_block = conv[(i*2):(i*2 + 2), (j*2):(j*2 + 2)]
                # 选取最大值
                res[i, j] = np.amax(conv_block, axis=(0,1))
        return res
    
    def backprop(self, d_L_d_out):
        d_L_d_input = np.zeros(self.input.shape)

        for i in range(self.pool_height):
            for j in range(self.pool_width):
                # 获取2*2conv块 这里得到的是2*2*8
                conv_block = self.input[(i*2):(i*2 + 2), (j*2):(j*2 + 2)]
                # 获取最大值的一列
                amax = np.amax(conv_block, axis=(0, 1))

                h, w, f = conv_block.shape

                for i2 in range(h):
                    for j2 in range(w):
                        for f2 in range(f):
                            if conv_block[i2, j2, f2] == amax[f2]:
                                d_L_d_input[i * 2 + i2, j * 2 + j2, f2] = d_L_d_out[i, j, f2]

        return d_L_d_input
