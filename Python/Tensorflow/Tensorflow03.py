from tensorflow.examples.tutorials.mnist import input_data as inp
import matplotlib.pyplot as plt

# tensorflow mnist
# --------------------------------------------------------------------------------
# input_data 내의 data를 one_hot으로 가져온다. / mnet/data에서 가져온다.
data = inp.read_data_sets('../mnet/data/', one_hot=True)
# print('학습용 데이터', data.train) # Extracting ( 적출하다. )
# print('검증용 데이터', data.test)

# num_examples 데이터갯수, images 이미지, labels 정답
# --------------------------------------------------------------------------------
# num_examples - data의 sample 데이터의 갯수를 가져온다.
# print('학습용 데이터 갯수',data.train.num_examples) # 55000
# print('검증용 데이터 갯수',data.test.num_examples)  # 10000

# 학습용 100번째 데이터 - Tensorflow03_sample.py 보기
# 학습용 100번째 데이터 실제 이미지 & 정답 - Tensorflow03_sample.py
# 학습용 1000번째 데이터 실제 이미지 & 정답 - Tensorflow03_sample.py

# 행렬을 나타내는 듯하다 확인 필요.
# print('이미지 데이터 차원', data.train.images[99].shape)

# labels 정답
# print('학습용 100번째 데이터 정답', data.train.labels[99])
# print('검증용 1000번째 데이터',data.test.images[999])
# ================================================================================


# 확인하기 ( 잘 모르겠음 데이터 분석 중 )
# --------------------------------------------------------------------------------
import tensorflow as tf
import numpy as np

data = np.loadtxt("..\\data\\cars.csv", skiprows=1, delimiter=",", unpack=True)

x = tf.placeholder(tf.float32,[None])
y = tf.placeholder(tf.float32,[None])

w = tf.Variable(tf.random_normal([1]))
b = tf.Variable(tf.random_normal([1]))

h = w * x + b

cost = tf.reduce_mean(tf.square(y-h))

learning_rate = 0.1     #nan
learning_rate = 0.01    #nan
# learning_rate=0.001   #cost=244.00842
# learning_rate=0.0001   #cost=261.67618
# learning_rate=0.0005   #cost=251.33484
# learning_rate=0.0003   #253.38168
# learning_rate=0.003    #229.81349
# learning_rate = 0.0035     #228.9206

train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # for i in range(2001):
    #     sess.run(train,feed_dict={x:data[0],y:data[1]})   #훈련
    #     if i%200==0:
    #         print(sess.run([cost,w,b],feed_dict={x:data[0],y:data[1]}))
    # print('w=',sess.run(w),'b=',sess.run(b))
#     #속도가 10일때의 제동거리를 예측하시오
#     print('속도가 10일때의 제동거리=',sess.run(h,feed_dict={x:[10]}))
#     #속도가 5,30일때의 제동거리를 예측하시오
    result=sess.run(h, feed_dict={x: [5, 30]})
    # print('속도가 5,30일때의 제동거리=',result)
    plt.plot([5,30],[result[0],result[1]])
    # plt.show()

# --------------------------------------------------------------------------------
# setosa     --> 1,0,0
# versicolor --> 0,1,0
# virginica  --> 0,0,1
# data=np.loadtxt('data\\iris1.csv',delimiter=',',skiprows=1)
# print(data)
# print('데이터=',data[:,0:4])
# print('정답=',data[:,4:])

# ================================================================================