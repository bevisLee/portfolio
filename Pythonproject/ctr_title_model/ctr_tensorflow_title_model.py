

# 참조 - https://github.com/kkb2849/tensorflow-models/tree/master/title_ctr_model


import tensorflow as tf
import re
import collections
import functools

#file load

f = open('C:/Users/bevis/Downloads/study_ctr_title_model/content_list.csv','r', encoding='UTF8')
line_counter = 0
unsupport_title_counter = 0
title_arr = []
ctr_arr = []

ex_hangul = re.compile('[^ ㄱ-ㅣ가-힣]+') # 한글과 띄어쓰기를 제외한 모든 글자

def map_hangul(str):
    str = str.replace('_',' ')
    str = ex_hangul.sub('',str)
    str = str.replace('  ',' ')
    return str

while True:    
    line = f.readline()
    if not line: break
    line_counter += 1
    
    line = line.replace('""',' ')
    content_info = line.split('"')
    
    try:
        title = map_hangul(content_info[1])
        ctr = content_info[2].split(',')[1]
        title_arr.append(title)
        ctr_arr.append([float(ctr)])
        if line_counter % 10000 == 0:
            print(line_counter, ":", title, ctr)
    except:
        unsupport_title_counter += 1
        
print('총 라인 수 : ', line_counter)
print('타이틀 수 : ', len(title_arr))
print('CTR 수 : ', len(ctr_arr))
print('에러 수 : ',unsupport_title_counter)
    
print(title_arr[:5])
print(ctr_arr[:5])

import matplotlib.pyplot as plt
import random

random_x = []
for i in range(len(ctr_arr)):
    random_x.append(random.randrange(1,10000))

plt.figure(num=None, figsize=(20, 6), dpi=80, facecolor='w', edgecolor='k')
plt.plot(ctr_arr, random_x, 'ro', alpha=0.01)
plt.legend()
plt.show()

grade = [0,0]
ctr_grade_arr = []
for ctr in ctr_arr:
    if ctr[0] < 0.015:
        grade[0] += 1
        ctr_grade_arr.append([1,0])
    else:
        grade[1] += 1
        ctr_grade_arr.append([0,1])

print(grade)
print(ctr_grade_arr[:5])

#dictionary를 만들기위해 단어만 주욱 모은다
words = functools.reduce(lambda x,y: x+' '+y, title_arr).split(' ')
print(len(words))
print(words[:10])

vocabulary_size = 20000

count = [['UNK', -1]] #unknown
count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
print(len(count))
print(count[:20])

#dictionary 정의 & feeding
dictionary = dict()
for word, _ in count:
    dictionary[word] = len(dictionary) #여기서 word는 count를 돌린거라서 유니크함

print(len(dictionary))

unk_count = 0

#unknown counting
for word in words:
    if word not in dictionary:
        unk_count += 1
    
#title_arr 를 index값으로 title_vector_arr로 변환
title_vector_arr = []
for title in title_arr:
    title_vector = []
    word_arr = title.split(' ')
    for word in word_arr:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
        title_vector.append(index)
    title_vector_arr.append(title_vector)
    
count[0][1] = unk_count
reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

# del words  # Hint to reduce memory.
print('Most common words (+UNK)', count[:5])
print('Title vector array sample', title_vector_arr[:10])
for title_vector in title_vector_arr[:10]:
    print('Reverse dict', [reverse_dictionary[word_index] for word_index in title_vector])

print(len(title_vector_arr))

# max_length = 0
# for title_vector in title_vector_arr:
#     if max_length < len(title_vector):
#         max_length = len(title_vector)
# print(max_length)

max_length = 20

#vector 사이즈를 정의하고, 부족한 부분은 empty_index로 채워서 사이즈를 맞춘다
empty_index = 0
title_vector_size = max_length
index = 0
for title_vector in title_vector_arr:
    while(len(title_vector) < title_vector_size):
        title_vector.append(empty_index)
    title_vector_arr[index] = title_vector[0:max_length]
    index +=1

print('Title vector array sample', title_vector_arr[:10])

import random

def generate_batch(title_vector_arr, ctr_grade_arr, batch_size):
    total_size = len(title_vector_arr)
    index_arr = [int(total_size*random.random()) for i in range(100)]
    batch_title = []
    batch_label = []
    for i in index_arr:
        batch_title.append(title_vector_arr[i])
        batch_label.append(ctr_grade_arr[i])
    return batch_title, batch_label

def xavier_init(n_inputs, n_outputs):
    init_range = tf.sqrt(6.0/(n_inputs+n_outputs))
    return tf.random_uniform_initializer(-init_range, init_range)

grade_size = len(grade)

X = tf.placeholder(tf.float32, shape=(None, title_vector_size), name='title_vector_arr')
Y = tf.placeholder(tf.float32, shape=(None, grade_size), name='ctr_grade_arr')

neuron_width = 1000

with tf.name_scope('hidden1') as scope:
    W1 = tf.get_variable(shape=[title_vector_size, neuron_width], initializer=xavier_init(title_vector_size, neuron_width), name='hidden1')
    b1 = tf.Variable(tf.zeros([neuron_width]), name='bias1')

with tf.name_scope('hidden2') as scope:
    W2 = tf.get_variable(shape=[neuron_width, neuron_width], initializer=xavier_init(neuron_width, neuron_width), name='hidden2')
    b2 = tf.Variable(tf.zeros([neuron_width]), name='bias2')

with tf.name_scope('hidden3') as scope:
    W3 = tf.get_variable(shape=[neuron_width, neuron_width], initializer=xavier_init(neuron_width, neuron_width), name='hidden3')
    b3 = tf.Variable(tf.zeros([neuron_width]), name='bias3')

with tf.name_scope('hidden4') as scope:
    W4 = tf.get_variable(shape=[neuron_width, neuron_width], initializer=xavier_init(neuron_width, neuron_width), name='hidden4')
    b4 = tf.Variable(tf.zeros([neuron_width]), name='bias4')

with tf.name_scope('hidden5') as scope:
    W5 = tf.get_variable(shape=[neuron_width, neuron_width], initializer=xavier_init(neuron_width, neuron_width), name='hidden5')
    b5 = tf.Variable(tf.zeros([neuron_width]), name='bias5')

with tf.name_scope('hidden6') as scope:
    W6 = tf.get_variable(shape=[neuron_width, neuron_width], initializer=xavier_init(neuron_width, neuron_width), name='hidden6')
    b6 = tf.Variable(tf.zeros([neuron_width]), name='bias6')

with tf.name_scope('hidden7') as scope:
    W7 = tf.get_variable(shape=[neuron_width, grade_size], initializer=xavier_init(neuron_width, grade_size), name='hidden7')
    b7 = tf.Variable(tf.zeros([grade_size]), name='bias7')

learning_rate = 0.1
print(X)
print(W1)
print(b1)
L2 = tf.nn.softmax(tf.matmul(X, W1) + b1)
L3 = tf.nn.softmax(tf.matmul(L2, W2) + b2)
# L4 = tf.nn.relu(tf.matmul(L3, W3) + b3)
# L5 = tf.nn.relu(tf.matmul(L4, W4) + b4)
# L6 = tf.nn.relu(tf.matmul(L5, W5) + b5)
# L7 = tf.nn.relu(tf.matmul(L6, W6) + b6)
h = tf.nn.softmax(tf.matmul(L3, W7) + b7)
# h = tf.matmul(L2, W3) + b3

with tf.name_scope('cross_entropy'):
#     cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h, labels=Y, name='cost'))
    cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(h), reduction_indices=1))
    tf.summary.scalar('cost', cost)

with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(h,1), tf.argmax(Y,1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        tf.summary.scalar('accuracy', accuracy)
        
merged = tf.summary.merge_all()

batch_size = 1000

acc_arr = []
step_arr = []

init = tf.global_variables_initializer()

sample_cost = 0
with tf.Session() as sess:
    train_writer = tf.summary.FileWriter('tensorboard/train',sess.graph)
    sess.run(init)
    for step in range(20000):
        batch_title, batch_ctr = generate_batch(title_vector_arr, ctr_grade_arr, batch_size)
        tb_summary, _ = sess.run([merged,optimizer], feed_dict={X: batch_title, Y: batch_ctr})
        sample_cost = sess.run(cost, feed_dict={X: batch_title, Y: batch_ctr})
        acc= sess.run(accuracy, feed_dict={X: batch_title, Y: batch_ctr})
        h_sample= sess.run(h, feed_dict={X: batch_title, Y: batch_ctr})
        train_writer.add_summary(tb_summary, step)
        if step % 100 == 0:
            train_writer.add_summary(tb_summary, step)
            acc_arr.append(acc)
            step_arr.append(step)
            print(step,':', acc)
            print(h_sample[1])
            print('-------------------')

import matplotlib.pyplot as plt

plt.plot(step_arr, acc_arr, '-o')
plt.legend()
plt.show()

