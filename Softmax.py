import numpy as np

class Softmax:
    # inputNodes_len为池化层拉平后的长度 outputNodes_len为输出层节点个数
    def __init__(self, inputNodes_len, outputNodes_len):
        # 权重矩阵
        self.weights = np.random.randn(inputNodes_len, outputNodes_len) / inputNodes_len
        # 每个节点的偏差值
        self.biases = np.zeros(outputNodes_len)

    # 接收池化层输入
    def forward(self, pool_input):
        # 3维转1维 
        input = pool_input.flatten()
        # 获取权重相乘＋偏差结果
        ans = np.dot(input, self.weights) + self.biases
        # e运算
        exp = np.exp(ans)

        # 记录exp
        self.last_exp   = exp
        # 记录exp的和
        self.last_S     = np.sum(exp, axis=0)
        # 记录input
        self.last_input = input
        # 记录pool_input规格
        self.last_input_shape = pool_input.shape
        # 算得比例
        return exp / self.last_S
    
    # d_L_d_out = 对L偏导out
    def backprop(self, d_L_d_out, learn_rate):
        for i, gradient in enumerate(d_L_d_out):
            # 根据数学公式，在d_L_d_out中仅有一个元素不为0，我们只需要改变不为0的那一层即可
            if gradient == 0:
                continue
            
            exp = self.last_exp # e^ans
            S   = self.last_S   # exp的和

            # 初始化为k≠c，之后单独改变k=c的情况就可以了
            d_out_d_t    = -exp[i] *  exp / (S ** 2)
            d_out_d_t[i] =  exp[i] * (S - exp[i]) / (S ** 2)

            d_t_d_w = self.last_input
            d_t_d_b = 1

            d_t_d_inputs = self.weights
            d_L_d_t = gradient * d_out_d_t

            # np.newaxis 用于增加矩阵维度, .T用于倒置 ，@与dot.(a, b)相等
            d_L_d_w = d_t_d_w[np.newaxis].T @ d_L_d_t[np.newaxis]
            d_L_d_b = d_L_d_t * d_t_d_b
            d_L_d_inputs = d_t_d_inputs @ d_L_d_t

            # 根据梯度与学习率来更改 权重和偏差
            self.weights -= learn_rate * d_L_d_w
            self.biases  -= learn_rate * d_L_d_b

            return d_L_d_inputs.reshape(self.last_input_shape)



            





