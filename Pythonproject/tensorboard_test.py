
import tensorflow as tf
 
a=tf.constant(6,name="a")
b=tf.constant(3,name="b")
c=tf.constant(10,name="c")
d=tf.constant(5,name="d")
 
mul=tf.multiply(a,b,name="mul")
div=tf.div(c,d,name="div")
 
addn=tf.add_n([mul,div],name="addn")
 
sess=tf.Session()
output =sess.run(addn)
print(output)
writer=tf.summary.FileWriter('./visual',sess.graph)
writer.close()
sess.close()

writer=tf.summary.FileWriter('./visual',sess.graph)
writer.close()

tensorboard --logdir=C:\Python36\visual #cmd에서 실행




