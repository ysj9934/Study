# 지도학습 - 분류, 회귀
# 노션/윤성재에서 개인적으로 정리 해논거 확인 : https://www.notion.so/86ca2f45d8f24d55bcdc2a1f21a8c84b


# h = wx + b
# x = [1,2,3] / y = [1,2,3] 일때 최저 cost를 만드는 w와 b는?
# --------------------------------------------------------------------------------
# # 1) w = 0.5, b = 0
# print("w = 0.5, b = 0인 경우 ",((1-0.5*1+0)**2+(2-0.5*2+0)**2+(3-0.5*3+0)**2)/3)
# # 2) w = 0, b = 2
# print("w = 0, b = 2인 경우 ",((1-0*1+2)**2+(2-0*2+2)**2+(3-0*3+2)**2)/3)
# # 3) w = 1, b = 0.5 (최저)
# print("w = 1, b = 0.5인 경우 ",((1-1*1+0.5)**2+(2-1*2+0.5)**2+(3-1*3+0.5)**2)/3)
# # 4) w = 1, b = 1
# print("w = 1, b = 1인 경우 ",((1-1*1+1)**2+(2-1*2+1)**2+(3-1*3+1)**2)/3)
# ================================================================================

# 확인하기 ( 잘 모르겠음 데이터 분석 중 )
# --------------------------------------------------------------------------------
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
trainx = [1,2,3]   #학습데이터
trainy = [1,2,3]

w = tf.Variable(tf.random_normal([1]))
b = tf.Variable(tf.random_normal([1]))

h = w * trainx + b

cost = tf.reduce_mean(tf.square(h-trainy))

train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())  # 변수 초기화
#     for i in range(2001):
#         sess.run(train)     # 학습
#         if i%100==0:
#             print(i,sess.run(cost),sess.run(w),sess.run(b))
# ================================================================================

# 확인하기 ( 잘 모르겠음 데이터 분석 중 )
# --------------------------------------------------------------------------------
# x = tf.placeholder(데이터타입,shape,name)

x = tf.placeholder(tf.float32,[3,1])
x = tf.placeholder(tf.float32,[3])

x = tf.placeholder(tf.float32,[None])
y = tf.placeholder(tf.float32,[None,1])

# with tf.Session() as sess:
#     print(sess.run(x,feed_dict={x:[[1],[2],[3]]}))
#     print(sess.run(x,feed_dict={x:[1,2,3]}))
#     print(sess.run(x,feed_dict={x:[1,2,3,5,4,6]}))
#     print(sess.run(x,feed_dict={x:[1]}))
#     print(sess.run(y,feed_dict={y:[[1]]}))
#     print(sess.run(y,feed_dict={y:[[1],[2],[3]]}))
# ================================================================================

# 확인하기 ( 잘 모르겠음 데이터 분석 중 )
# --------------------------------------------------------------------------------
x = tf.placeholder(tf.float32,shape=[None])  #데이터
y = tf.placeholder(tf.float32,shape=[None])  #정답

w = tf.Variable(tf.random_normal([1]))
b = tf.Variable(tf.random_normal([1]))

h = w * x + b

cost = tf.reduce_mean(tf.square(y-h))

train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for i in range(2001):
#         # result=sess.run([train,cost,w,b],feed_dict={x:[1,2,3],y:[3,6,8]})
#         result = sess.run([train,cost,w,b],feed_dict={x:[1,2,3,4,5],
#                                                     y:[2.6,8.7,9.1,12.5,13]})
#         if i%200==0:
#             print(i, result[0],result[1],result[2],result[3])
# ================================================================================
