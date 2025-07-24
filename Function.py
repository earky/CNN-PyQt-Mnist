import mnist
import numpy as np
import Pool
import Softmax
import Conv33
import os
import gzip
import logging

def parse_mnist(minst_file_addr: str = None) -> np.array:
    """解析MNIST二进制文件, 并返回解析结果
    输入参数:
        minst_file: MNIST数据集的文件地址. 类型: 字符串.

    返回值:
        解析后的numpy数组
    """
    if minst_file_addr is not None:
        minst_file_name = os.path.basename(minst_file_addr)  # 根据地址获取MNIST文件名字
        with gzip.open(filename=minst_file_addr, mode="rb") as minst_file:
            mnist_file_content = minst_file.read()
        if "label" in minst_file_name:  # 传入的为标签二进制编码文件地址
            data = np.frombuffer(buffer=mnist_file_content, dtype=np.uint8, offset=8)  # MNIST标签文件的前8个字节为描述性内容，直接从第九个字节开始读取标签，并解析
        else:  # 传入的为图片二进制编码文件地址
            data = np.frombuffer(buffer=mnist_file_content, dtype=np.uint8, offset=16)  # MNIST图片文件的前16个字节为描述性内容，直接从第九个字节开始读取标签，并解析
            data = data.reshape(-1, 28, 28)
    else:
        logging.warning(msg="请传入MNIST文件地址!")

    return data


# test_images = parse_mnist(minst_file_addr="t10k-images-idx3-ubyte.gz")  # 训练集图像
# test_labels = parse_mnist(minst_file_addr="t10k-labels-idx1-ubyte.gz")  # 训练集标签
# train_images = parse_mnist(minst_file_addr="t10k-images-idx3-ubyte.gz")  # 训练集图像
# train_labels = parse_mnist(minst_file_addr="t10k-labels-idx1-ubyte.gz")  # 训练集标签

conv = Conv33.Conv33(8)
pool = Pool.Pool()
softmax = Softmax.Softmax(13*13*8, 10)

def forward(image, label):
    out = conv.forward((image / 255) - 0.5)
    out = pool.forward(out)
    out = softmax.forward(out)

    loss = -np.log(out[label])
    
    acc = 1 if np.argmax(out) == label else 0

    return out, loss, acc

def train(im, label, lr=.005):
    # Forward
    out, loss, acc = forward(im, label)

    # 初始化梯度
    gradient = np.zeros(10)
    gradient[label] = -1 / out[label]
 
    # 反向传播
    gradient = softmax.backprop(gradient, lr)
    gradient = pool.backprop(gradient)
    gradient = conv.backprop(gradient, lr)
 
    return loss, acc

# print('MNIST CNN initialized!')

# loss = 0
# num_correct = 0
# for epoch in range(3):
#     print('--- Epoch %d ---' % (epoch + 1))
 
#     # Shuffle the training data
#     permutation = np.random.permutation(len(train_images))
#     train_images = train_images[permutation]
#     train_labels = train_labels[permutation]
 
#     # Train!
#     loss = 0
#     num_correct = 0
 
#     # i: index
#     # im: image
#     # label: label
#     for i, (im, label) in enumerate(zip(train_images, train_labels)):
#         if i > 0 and i % 100 == 99:
#             print(
#                 '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
#                 (i + 1, loss / 100, num_correct)
#             )
#             loss = 0
#             num_correct = 0
 
#         l, acc = train(im, label)
#         loss += l
#         num_correct += acc
 
# # Test the CNN
# print('\n--- Testing the CNN ---')
# loss = 0
# num_correct = 0
# for im, label in zip(test_images, test_labels):
#     _, l, acc = forward(im, label)
#     loss += l
#     num_correct += acc
 
# num_tests = len(test_images)
# print('Test Loss:', loss / num_tests)
# print('Test Accuracy:', num_correct / num_tests)