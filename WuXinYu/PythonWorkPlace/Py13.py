import gzip
import paddle.v2 as paddle

def softmax_regression(img):
    predict = paddle.layer.fc(input=img,
                              size=10,
                              act=paddle.activation.Softmax())
    return predict

def multilayer_perceptron(img):
    # 第一个全连接层，激活函数为ReLU
    hidden1 = paddle.layer.fc(input=img, size=128, act=paddle.activation.Relu())
    # 第二个全连接层，激活函数为ReLU
    hidden2 = paddle.layer.fc(input=hidden1,
                              size=64,
                              act=paddle.activation.Relu())
    # 以softmax为激活函数的全连接输出层，输出层的大小必须为数字的个数10
    predict = paddle.layer.fc(input=hidden2,
                              size=10,
                              act=paddle.activation.Softmax())
    return predict

ef convolutional_neural_network(img):
    # 第一个卷积-池化层
    conv_pool_1 = paddle.networks.simple_img_conv_pool(
        input=img,
        filter_size=5,
        num_filters=20,
        num_channel=1,
        pool_size=2,
        pool_stride=2,
        act=paddle.activation.Relu())
    # 第二个卷积-池化层
    conv_pool_2 = paddle.networks.simple_img_conv_pool(
        input=conv_pool_1,
        filter_size=5,
        num_filters=50,
        num_channel=20,
        pool_size=2,
        pool_stride=2,
        act=paddle.activation.Relu())
    # 以softmax为激活函数的全连接输出层，输出层的大小必须为数字的个数10
    predict = paddle.layer.fc(input=conv_pool_2,
                              size=10,
                              act=paddle.activation.Softmax())
    return predict

# 该模型运行在单个CPU上
paddle.init(use_gpu=False, trainer_count=1)

images = paddle.layer.data(
    name='pixel', type=paddle.data_type.dense_vector(784))
label = paddle.layer.data(
    name='label', type=paddle.data_type.integer_value(10))

# predict = softmax_regression(images) # Softmax回归
# predict = multilayer_perceptron(images) #多层感知器
predict = convolutional_neural_network(images) #LeNet5卷积神经网络

cost = paddle.layer.classification_cost(input=predict, label=label)

parameters = paddle.parameters.create(cost)

optimizer = paddle.optimizer.Momentum(
    learning_rate=0.1 / 128.0,
    momentum=0.9,
    regularization=paddle.optimizer.L2Regularization(rate=0.0005 * 128))

trainer = paddle.trainer.SGD(cost=cost,
                             parameters=parameters,
                             update_equation=optimizer)

from paddle.v2.plot import Ploter

train_title = "Train cost"
test_title = "Test cost"
cost_ploter = Ploter(train_title, test_title)

step = 0

# event_handler to plot a figure
def event_handler_plot(event):
    global step
    if isinstance(event, paddle.event.EndIteration):
        if step % 100 == 0:
            cost_ploter.append(train_title, step, event.cost)
            cost_ploter.plot()
        step += 1
    if isinstance(event, paddle.event.EndPass):
        # save parameters
        with gzip.open('params_pass_%d.tar.gz' % event.pass_id, 'w') as f:
            parameters.to_tar(f)

        result = trainer.test(reader=paddle.batch(
            paddle.dataset.mnist.test(), batch_size=128))
        cost_ploter.append(test_title, step, result.cost)

lists = []

def event_handler(event):
    if isinstance(event, paddle.event.EndIteration):
        if event.batch_id % 100 == 0:
            print "Pass %d, Batch %d, Cost %f, %s" % (
                event.pass_id, event.batch_id, event.cost, event.metrics)
    if isinstance(event, paddle.event.EndPass):
        # save parameters
        with gzip.open('params_pass_%d.tar.gz' % event.pass_id, 'w') as f:
            parameters.to_tar(f)

        result = trainer.test(reader=paddle.batch(
            paddle.dataset.mnist.test(), batch_size=128))
        print "Test with Pass %d, Cost %f, %s\n" % (
            event.pass_id, result.cost, result.metrics)
        lists.append((event.pass_id, result.cost,
                      result.metrics['classification_error_evaluator']))
trainer.train(
    reader=paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.mnist.train(), buf_size=8192),
        batch_size=128),
    event_handler=event_handler_plot,
    num_passes=5)
训练过程是完全自动的，event_handler里打印的日志类似如下所示：

# Pass 0, Batch 0, Cost 2.780790, {'classification_error_evaluator': 0.9453125}
# Pass 0, Batch 100, Cost 0.635356, {'classification_error_evaluator': 0.2109375}
# Pass 0, Batch 200, Cost 0.326094, {'classification_error_evaluator': 0.1328125}
# Pass 0, Batch 300, Cost 0.361920, {'classification_error_evaluator': 0.1015625}
# Pass 0, Batch 400, Cost 0.410101, {'classification_error_evaluator': 0.125}
# Test with Pass 0, Cost 0.326659, {'classification_error_evaluator': 0.09470000118017197}