import tensorflow as tf
import numpy as np

# 변수 ( Variable ) & 오퍼레이션 ( operation ) 예제 1
# --------------------------------------------------------------------------------
a = tf.constant(3)
b = tf.constant(2)
c = tf.constant(7)

d = tf.multiply(a,b) # 6
e = tf.add(c,b) # 9
f = tf.subtract(d,e) # -3

# with tf.Session() as sess:
    # print(sess.run(f))
    # print(sess.run([a,b,c,d,e,f]))

# 변수 tf.Variable()-초기화 작업후에 사용
a = tf.constant(3)
b = tf.constant(5)

c = tf.Variable(0) # 변수 0으로 초기화

op1 = a+b

op2 = tf.assign(c,op1)   # assign - 세션 속에서 op1의 값을 변수 c에 저장

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())   # 변수초기화
#     print(sess.run([a,b,c]))
#     print(sess.run([op2,c]))
# --------------------------------------------------------------------------------

# 변수 ( Variable ) & 플레이스홀더 ( placeholder ) 예제 2
# --------------------------------------------------------------------------------
state = tf.Variable(0)
one = tf.constant(1)

new_value = tf.add(state,one) # 1
update = tf.assign(state,new_value) # 1

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for x in range(5): # 1 ~ 5
#         sess.run(update)
#         res = sess.run(state)
#         print(res)

# 변수 tf.placeholder() - 실행시에 값을 넣고 진행
a = tf.constant(5)
b = tf.placeholder(tf.int32)

op1 = a*b

# with tf.Session() as sess:
#     print(sess.run(op1,feed_dict={b:10})) # 변수 속에 b의 값을 넣어준다.
# --------------------------------------------------------------------------------

# 구구단 예제
# --------------------------------------------------------------------------------
dan = tf.placeholder(tf.int32)  # 단수
i = tf.placeholder(tf.int32)    # 값 / 1-9

op1 = tf.multiply(dan,i)   # 답
# op1 = dan * i

# with tf.Session() as sess:
#     for k in range(1,10): # i의 값 넣기
#         # print(sess.run(op1, feed_dict={dan:7,i:k}))
#         res = sess.run([dan,i,op1],feed_dict={dan:7,i:k}) # 7단
#         # print(res)   #[array(7), array(1), 7]
#         # print(res[0],type(res[0]))   # 7 (단수) <class 'numpy.ndarray'>
#         print('{} X {} = {}'.format(res[0],res[1],res[2]))
# --------------------------------------------------------------------------------

# placeholder
# --------------------------------------------------------------------------------
a = tf.constant(7)
b = tf.placeholder(tf.int32,[3]) # 3개를 넣을 수 있다?

op1 = a*b

# with tf.Session() as sess:
#     print(sess.run(op1,feed_dict={b:[1,2,3]}))
# --------------------------------------------------------------------------------

# 예제
# --------------------------------------------------------------------------------
a = tf.constant(51)
b = tf.constant(10)
c = tf.constant(3)
# 변수 처리
v = tf.Variable(0) 

op1 = (a+b)*c
op2 = tf.assign(v,op1)

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())  # 변수초기화
#     res = sess.run([a,b,c,op1])  #[51, 10, 3, 183]
#     print('a =',res[0])
#     print('b =',res[1])
#     print('c =',res[2])
#     print('op1 =',res[3])
#     res = sess.run([a,b,c,v,op2])
#     print(res)
# --------------------------------------------------------------------------------

# 예제
# --------------------------------------------------------------------------------
i = tf.random_normal((1,5),0,1)

v = tf.Variable(i)

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(v))
# --------------------------------------------------------------------------------