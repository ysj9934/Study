가상환경_생성 = '''
Anaconda3-5.2.0-Windows-x86_64.exe

가상환경생성
File > Settings > Project > pythonInterpreter
Tensorflow > Install version 1.5.1 or 1.5.0
'''

import tensorflow as tf

# tensorflow 버전확인
chk = tf.__version__
# print(chk) # 1.5.1

# 상수노드 - tf.constant()
# --------------------------------------------------------------------------------
# 상수열의 문자형
sampleT = tf.constant('Hello, Tensorflow !')
sessT = tf.Session()
# print(sessT.run(sampleT))
# --------------------------------------------------------------------------------
# 상수 설정
n1 = tf.constant(3.14)
n2 = tf.constant(4,dtype=tf.float16)
n3 = tf.constant(5.0,name='c')
# print(n1) # Tensor("Const_1:0", shape=(), dtype=float32)
# print(n2) # Tensor("Const_2:0", shape=(), dtype=float16)
# print(n3) # Tensor("c:0", shape=(), dtype=float32)

# 그냥 상수 설정만 하면 정보가 나온다.
# --------------------------------------------------------------------------------
# 변수 표현하기
# 상수 정의(선언)
a = tf.constant(120, name="a") # name을 붙이면 별칭을 사용할 수 있다.
b = tf.constant(130, name="b")
c = tf.constant(140, name="c")
# 변수 정의
v = tf.Variable(0, name="v") # 변수 v에는 초기값은 0이다.
# 데이터 플로우 그래프 정의
calc_op = a + b + c
assign_op = tf.assign(v, calc_op) # a + b + c를 계산 후 v에 대입
# 세션 실행
sess = tf.Session()
sess.run(assign_op)
# v 출력하기
# print(sess.run(v))
# ================================================================================

# 세션 - Session()
# 성능을 향상 시키기 위해 정의와 실행을 분리한다.
# 노드(상수)가 실제 값을 가지려면 Session객체를 생성하여 실행해야 된다.
# --------------------------------------------------------------------------------

# 세션객체 생성
sess = tf.Session()

# 세션실행
# 위의 상수노드 파트에서 가져온다.
# print(sess.run(n1));print(sess.run(n2));print(sess.run(n3))
# print(sess.run([n1,n2,n3]))

# 세션종료
sess.close()
# ================================================================================
# operation
# 노드를 연결하는 역할을 한다.
# 수학적 함수를 이용하여 만들어지며 오퍼레이션 자체도 노드이다.
# --------------------------------------------------------------------------------

n1 = tf.constant(3.14)
n2 = tf.constant(5.6)
# n3 = tf.constant(2) # err
# n3=tf.constant(2,tf.float16) # err
n3=tf.constant(2,tf.float32)

# operation 사용 (사칙연산)
op1 = n1+n2
op2 = tf.subtract(n1,n2)
op3 = tf.multiply(n1,n2)
op4 = n1/n3
# sess=tf.Session()   #세션객체 생성
# print('더하기 = ',sess.run(op1))
# print('빼기 = ',sess.run(op2))
# print('곱하기 = ',sess.run(op3))
# print('나누기 = ',sess.run(op4))
sess.close()
# ================================================================================

# 플레이스홀더 - tf.placeholder
# 값을 넣을 공간을 만들어두는 기능이다.
# --------------------------------------------------------------------------------
# 플레이스홀더 정의
# a = tf.placeholder(tf.int32,[3])
# 정수 자료형 3개를 가진 배열
a = tf.placeholder(tf.int32,[None])
# 배열의 크기가 None이므로 원하는 크기의 배열을 지정하여 사용할 수 있다.
b = tf.constant(2)

op1 = a*b

sess = tf.Session()

r1 = sess.run(op1, feed_dict={a:[1,2,3]})
r2 = sess.run(op1, feed_dict={a:[10,20,10,40]})

# print(r1) # [2 4 6]
# print(r2) # [20 40 20 80]
# --------------------------------------------------------------------------------
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
op = x * y

# with tf.Session() as sess:
    # print(sess.run(op,feed_dict={x:3,y:10}))
    # 30.0 / float라 소수점이 나온 듯 하다.
    # print(sess.run(op,feed_dict={x:[1,2,3],y:5}))
    # [ 5. 10. 15.] / 여러개 사용 가능
    # print(sess.run(op,feed_dict={x:[[1,2,3],[4,5,6]], y:5}))
    # (3 x 2) 3행 2열
# --------------------------------------------------------------------------------
# random_normal ((행,렬),,) 중에 하나
x = tf.random_normal((1,4),10,1)
y = tf.placeholder(tf.float32)

op = x * y

# with tf.Session() as sess:
#     res = sess.run([x,y,op],feed_dict={y:10})
#     print('x =\n',res[0])
#     print('y =\n',res[1])
#     print('op =\n',res[2])

# ================================================================================