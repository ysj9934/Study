import tensorflow as tf
import numpy as np


# numpy와 사용하기
# --------------------------------------------------------------------------------
# h = xw + b

# np.random.randn(행,렬) : 랜덤 수를 (행,렬)에 맞게 나타낸다.
xdata = np.random.randn(5,10)
wdata = np.random.randn(10,1)
# placeholder - 값을 넣어둔다.
x = tf.placeholder(tf.float32,(5,10))
w = tf.placeholder(tf.float32,(10,1))
# fill((행,렬),num.) -
b = tf.fill((5,1),-5.)
# tf.matmul() - 행렬곱 함수
h = tf.matmul(x,w)+b

# with tf.Session() as sess:
#     print(sess.run(x,feed_dict={x:xdata}))
# # 행렬곱
# # x(5, 10)  X w(10,1) = (5,1)
#     print(sess.run(h,feed_dict={x:xdata,w:wdata}))

# ================================================================================

# 행렬 차수 변경
# --------------------------------------------------------------------------------
a = tf.constant([1,2,3,4,5,6,7,8])
b = tf.constant([[1,2],[3,4],[5,6],[7,8]])

# with tf.Session() as sess:
    # 한 줄로 구성되어 있는 상수를 (3,3)로 변경시킨다. / 갯수가 맞지 않는 행렬은 오류가 난다. [errcode: ValueError]
    # print(sess.run(tf.reshape(a,(3,3))))
    # 수가 맞지 않는 행렬은 오류가 난다. [errcode : ValueError]
    # print(sess.run(tf.reshape(b,(2,2,3))))
    # (2,2,2) 형태로 나오게 된다.
    # print(sess.run(tf.reshape(b,(2,-1,2))))
    # 1행으로 나오게 된다.
    # print(sess.run(tf.reshape(b,(-1,))))

reshape에_대하여 = '''
* 형변환 reshape

> 1차원 > 2차원 변환

a = np.arange(1,9)
b = np.reshape(a,(2,4)) 
# a의 값들을 2행 4열로 나눠진다.
res = [[1 2 3 4]
        [5 6 7 8]]

> 1차원 > 3차원 변환

a = np.arange(1,9)
b = a.reshape(2,2,2)
# a의 값들을 2행 2열로 2개씩 묶어 나오게 한다.
res = [[[1 2]
        [3 4]]

        [[5 6]
        [7 8]]]
        
> 행렬 위치에 -1 이 있는 경우

a = np.arange(12)
b = a.reshape(-1,3)
# 행 값에 -1이 있는 경우 열 값에 원하는 값을 넣으면 자동으로 행이 맞춰진다.
# 다만 맞는 경우에만 가능하다.
'''
# ================================================================================

# 0으로 구성된 행렬
# --------------------------------------------------------------------------------
# 0을 (2,3)으로 배치한다.
one = tf.zeros((2,3))
# 0을 이미 배치해준 상태이다. (2,4) / 안의 내용(값)은 0으로 치환된 듯 하다.
two = tf.zeros_like([[1,2,3,9],[4,5,6,8]])

# with tf.Session() as sess:
#     print(sess.run(one))
#     print(sess.run(two))
# ================================================================================

# 형변환 + 위의 내용 '0으로 구성된 행렬'
# --------------------------------------------------------------------------------
one = tf.constant([1.,2.,3.])
two = tf.constant([True,False,True,True])

# with tf.Session() as sess:
    # int형으로 변환된 one 값이 나온다.
    # print(sess.run(tf.cast(one,tf.int32)))

    # True는 1, False는 0으로 나온다.
    # print(sess.run(tf.cast(two,tf.int32)))  #형변환  [1 0 1 1]

    # int와 다르게 float는 뒤에 .을 찍어 본인이 float임을 나타닌다.
    # print(sess.run(tf.cast(two,tf.float32)))  #형변환 [1. 0. 1. 1.]

    # tf.reduce_mean() - 평균값 구하기
    # (int라 소수점이하는 나오지 않아 0이 나온 듯 하다)
    # print(sess.run(tf.reduce_mean(tf.cast(two,tf.int32))))  #0

    # (형태가 float라 소수점이하가 나오기 때문에 평균값 3/4 = 0.75로 제대로 나온다.
    # print(sess.run(tf.reduce_mean(tf.cast(two,tf.float32)))) #0.75
# ================================================================================

# argmax - tf.argmax()
# 가장 큰값을 찾아 인덱스를 반환하기
# 유사 방식으로 tf.argmin()이 존재한다.
# --------------------------------------------------------------------------------
a = tf.constant([3,2,4,1])
b = tf.constant([[3,1,5],[0,7,10],[5,4,3]])

with tf.Session() as sess:
    # print('a =', sess.run(a))
    # print('요소에 접근하기 위한 인덱스갯수 =', sess.run(tf.rank(a)))

    # 인덱스(index)를 반환하기 때문에 a에서 가장 큰 숫자인 4의 index인 2를 반환한다.
    print('가장 큰값의 위치 =', sess.run(tf.argmax(a)))


    # print('원핫으로 =', sess.run(tf.one_hot(a,5)))

# ================================================================================

# one_hot인코딩 - tf.argmax()
# 단 하나의 값만 True이고 나머지는 모두 False인 인코딩이다.
# 행렬을 자주사용하는 연산에서 스칼라값보다는 0,1로 구분된 행렬이 자주 이용된다.
# 원핫 : 데이터를 수 많은 0과 한개의 1의 값으로 데이터를 구분하는 방식이다.
# index가 5개인 경우 ( 0의 갯수는 4개, 1의 갯수는 1개이다. )
# 0: 10000
# 1: 01000
# 2: 00100
# 3: 00010
# 4: 00001
# --------------------------------------------------------------------------------

# print('b =\n',sess.run(b))
# print('요소에 접근하기 위한 인덱스갯수 =', sess.run(tf.rank(b)))

# 0은 열방향, 1은 행방향을 나타내며 순서대로 가장 큰 값을 반환한다.
# print('열방향 큰값의 위치 =', sess.run(tf.argmax(b,0)))
# print('행방향 큰값의 위치 =', sess.run(tf.argmax(b,1)))

# ================================================================================