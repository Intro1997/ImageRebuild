from ZSSR_by_myself.utils import *
import tensorflow as tf
import scipy.io as sio
import time

class ZSSR:
    # 两种处理情况
    REAL_IMAGE = {
        'img': "/img_kernel/charlie.png",
        'kernel': "img_kernel/charlie_0.mat",
        'noise_std' : 0.0125
    }
    KERNEL_IMAGE = {
        'img': "/img_kernel/BSD100_100_lr_rand_ker_c_X2.png",
        'kernel': "img_kernel/BSD100_100_lr_rand_ker_c_X2_0.mat",
        'noise_std' : 0.0
    }

    # 实际参数
    layer_width = 64                    # 每层的过滤器数量
    nn_depth = 8                        # 神经网络的深度
    init_variance = 0.1                 # 初始化方差，用于初始化过滤器
    filter_dimensions = (               # 每个隐藏层的过滤器维度
            [[3, 3, 3, layer_width]] +
            [[3, 3, layer_width, layer_width]] * (nn_depth - 2) +
            [[3, 3, layer_width, 3]])
    learning_rate = 0.001               # 学习率
    cost = None
    iter = None
    mse = None
    hr_father_source = None             # 输入的愿图数据
    iteration = 3000
    hr_father = None                    # 被裁剪出来的原图
    lr_son = None                       # 被下采样处理的 hr_father 图像
    train_output = None                 # 一次前向反向的输出
    aim_sacle_factor = [2, 2]           # 设置对输入图像的 scale_factor
    is_flip = True                      # 确定是否需要在测试中旋转图像，扩充训练集
    kernel = None                       # 用于获取 lr_son 的模糊核
    back_projection_iteration = 2       # 反投影的迭代次数
    noise_std = 0                       # 调整高斯噪声程度
    random_scale = False                # 是否使用随机缩放因子
    situation = None                    # 处理输入情况，分为 REAL_IMAGE 和 KERNEL_IMAGE


    # TF 参数
    learning_rate_ph = None             # 学习率
    lr_son_ph = None                    # hr 图像下采样后的 lr 图像
    hr_father_ph = None                 # crop 原图得到的 hr 图像
    cost_ph = None                      # tf 损失函数值
    train_op = None                     # tf 训练操作
    init_op = None                      # tf 初始化操作
    sess = None                         # tf session



    # 构造函数，初始化各种参数
    def __init__(self, situation, random_scale = False):
        if situation is 'REAL_IMAGE':
            self.situation = self.REAL_IMAGE
        elif situation is 'KERNEL_IMAGE':
            self.situation = self.KERNEL_IMAGE
        else:
            raise Exception('无对应 ' + str(situation) + ' 的处理情况，请检查输入值')

        self.random_scale = random_scale

        # 初始化输入图像
        self.input_image = self.situation['img'] if type(self.situation['img']) is not str else load_input_image(self.situation['img']) / 255.

        # 初始化 tensorflow 图
        self.model = tf.Graph()

        self.build_network()

        self.hr_father_source = self.input_image

        self.noise_std = self.situation['noise_std']

        self.init_sess()


    # 构建神经网络
    def build_network(self):
        with self.model.as_default():
            # 设置 placeholder
            self.lr_son_ph = tf.placeholder(tf.float32, name='lr_son')
            self.hr_father_ph = tf.placeholder(tf.float32, name='hr_father')
            self.learning_rate_ph = tf.placeholder(tf.float32, name='learning_rate')

            # 设置每一层的过滤器
            self.filters = [tf.get_variable('filter_%s' % (i), self.filter_dimensions[i],
                                            initializer=tf.random_normal_initializer(
                                                stddev=np.sqrt(
                                                    self.init_variance / np.prod(
                                                        self.filter_dimensions[i][0:3]
                                                    )
                                                )
                                            ))
                            for i in range(0, len(self.filter_dimensions))]

            # 计算有激活卷积
            forward_output = self.lr_son_ph
            for i in range(0, self.nn_depth - 1):
                forward_output = tf.nn.relu(
                    tf.nn.conv2d(forward_output, self.filters[i], strides=[1, 1, 1, 1], padding='SAME'))

            # 计算最后一层卷积
            forward_output = tf.nn.conv2d(forward_output, self.filters[self.nn_depth - 1], strides=[1, 1, 1, 1], padding='SAME')

            # 残差神经网络
            self.net_output_t = forward_output + self.lr_son_ph

            # 计算成本函数
            self.cost_t = tf.reduce_mean(tf.reshape(tf.abs(self.net_output_t  - self.hr_father_ph), [-1]))

            # 应用 ADAM 算法训练神经网络
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph).minimize(self.cost_t)

            # 对 tensoflow 的所有变量进行初始化
            self.init_op = tf.initialize_all_variables()

        print('build network')

    def init_sess(self):
        # 初始化变量
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.model, config=config)
        self.sess.run(self.init_op)

        self.cost = 0
        self.iter = 0
        self.mes = 0

    def forward_backword_process(self, lr_son, hr_father):
        # 获取和 hr_father 相同 shape 的 lr_son
        interpolated_lr_son = imresize(image=lr_son, is_show=False, output_shape=hr_father.shape, noise_std=self.noise_std)

        feed_dict = {'lr_son:0':np.expand_dims(interpolated_lr_son, 0),
                     'hr_father:0':np.expand_dims(hr_father, 0),
                     'learning_rate:0':self.learning_rate}

        _, self.cost, train_ouput = self.sess.run([self.train_op, self.cost_t, self.net_output_t], feed_dict)

        return train_ouput

    def forward_process(self, lr_son):
        interpolated_lr_son = imresize(lr_son, False, self.aim_sacle_factor, noise_std=self.noise_std)
        feed_dict = {'lr_son:0' : np.expand_dims(interpolated_lr_son, 0)}

        return np.squeeze(self.sess.run([self.net_output_t], feed_dict))


    def train(self):
        for self.iter in range(self.iteration):
            # 获取数据集合
            if self.iter % 5 == 0:
                print("iteration: %s" %(self.iter))
            # 获取原图随机裁剪

            if self.random_scale:
                scale = round(np.random.rand() * (1.0 - 0.5) + 0.5, 2)
            else:
                scale = 0.23
            self.hr_father = random_crop(self.hr_father_source, is_show=False, size = scale)
            # 获取裁剪的 LR 版本

            self.lr_son = imresize(self.hr_father, is_show=False, scale_factor=0.5, kernel=self.kernel, output_shape=None, noise_std=self.noise_std)

            # 一次训练输出
            self.train_output = self.forward_backword_process(self.lr_son, self.hr_father)

    # 将原图分别旋转至 0，90，180，270 度，并取得各个角度对应的垂直水平镜像
    # 将上述图像输入到神经网络中进行训练，取所有得到的输出结果的中位数
    def final_test(self):
        output = []
        for i in range(0, 1 + 7 * self.is_flip):
            test_input = np.rot90(self.input_image, i) if i < 4 else np.fliplr(np.rot90(self.input_image, i))

            tmp_output = self.forward_process(test_input)

            tmp_output = np.rot90(tmp_output, -i) if i < 4 else np.rot90(np.fliplr(tmp_output), -i)

            # 做两次反向投影，根据这种错误反馈机制得到更好的结果。
            for i in range(self.back_projection_iteration):
                tmp_output = back_projection(tmp_output, lr_img=self.input_image, down_kernel=self.kernel, up_kernel='cubic')

            output.append(tmp_output)

        # 取中位数
        final_sr = np.median(output, 0)

        for i in range(self.back_projection_iteration):
            final_sr = back_projection(final_sr , lr_img=self.input_image, down_kernel=self.kernel,
                                       up_kernel='cubic')

        return final_sr


    def run(self):
        self.kernel = self.situation['kernel'] if self.situation['kernel'] is None else sio.loadmat(self.situation['kernel'])['Kernel']

        self.init_sess()

        self.train()

        final = self.final_test()

        return final


tic = time.time()
situation = 'REAL_IMAGE'    # 可调整参数为：'REAL_IMAGE' 和 'KERNEL_IMAGE'
t = ZSSR(situation)
sr_output = t.run()
plt.imsave(situation + '_result.png', np.array(sr_output * 255, dtype='int'))
toc = time.time()
print('total time is :' + str(toc-tic))
