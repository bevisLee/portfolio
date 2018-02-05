import tensorflow as tf
import bage_utils.watch_util
import numpy as np
import pandas as pd
import os
import sys
import math


## python36/Lib/ 폴더 복사하여 처리
from bage_utils.date_util import DateUtil
from bage_utils.num_util import NumUtil
from bage_utils.timer_util import TimerUtil
from bage_utils.watch_util import WatchUtil

## python36/Lib/ 폴더 복사하여 처리
from nlp4kor.config import log, TENSORBOARD_LOG_DIR # cinfig.py 경로 재설정 필요

import traceback

def add(x_data):
    y_data = np.sum(x_data, axis=1)  # sum = add all
    return np.expand_dims(y_data, axis=1)
    # pass


def average(x_data):
    y_data = np.average(x_data, axis=1)
    return np.expand_dims(y_data, axis=1)

def multiply(x_data) :
    y_data = np.prod(x_data, axis=1) # multiply 함수보다 유용하여 추천(x_data 여러개라도 상관없음)
    # array([[1, 2],
    #        [3, 4]])
    # axis=0 [1*3, 2*4]  ↓ , axis=1 [1*2, 3*4]  → 
    print('y_data',y_data.shape)
    return np.expand_dims(y_data, axis=1) # tensor에 dimension 추가

def build_graph(scope_name, n_features, n_hiddens, n_classes, learning_rate, optimizer=tf.train.AdamOptimizer, activation=tf.tanh, weights_initializer=tf.truncated_normal_initializer, bias_value=0.0):
    x = tf.placeholder(tf.float32, [None, n_features])
    y = tf.placeholder(tf.float32, [None, n_classes])

    w1 = tf.get_variable(initializer=weights_initializer(), shape=[n_features, n_hiddens], name='w1') # tf.Variable 가 아니라 get_variable 사용해야함
    b1 = tf.get_variable(initializer=tf.constant(bias_value, shape=[n_hiddens]), name='b1')
    h1 = tf.nn.xw_plus_b(x, w1, b1)
    h1_out = activation(h1)  # 가운데가 1개

    w2 = tf.get_variable(initializer=weights_initializer(), shape=[n_hiddens, n_classes], name='w2')
    b2 = tf.get_variable(initializer=tf.constant(bias_value, shape=[n_classes]), name='b2')
    h2 = tf.nn.xw_plus_b(h1_out, w2, b2)
    y_hat = h2

    cost = tf.reduce_mean(tf.square(y - y_hat), name='cost')
    train_step = optimizer(learning_rate=learning_rate, name='optimizer').minimize(cost, name='train_step')

    rsme = tf.sqrt(cost, name='rsme')
    with tf.name_scope(scope_name):
        cost_ = tf.summary.scalar(tensor=cost, name='cost')
        summary = tf.summary.merge([cost_])
    return x, y, y_hat, cost, rsme, train_step, summary

    #    print(train_step)
    #    print(cost)
    #    print(cost_)
    #    exit()
    
if __name__ == '__main__' :
   os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # ignore tensorflow warnings
   tf.logging.set_verbosity(tf.logging.ERROR) # ignore tensorflow info

    func = multiply
    #   print(func.__name__)  # func name  확인
    n_features = 2 # x1, x2
    n_classes = 1 # y
    digits = list(range(-99,100,1))
    print(digits)
    n_train, n_test = 4000, 100 # 1% of 200 * 200

    x_data = np.random.choice(digits, (n_train+n_test, n_features), replace = True) # replace = True 뽑은 숫자를 다시 뽑도록 설정
    # print('x_data', x_data.shape) # (4100, 2) 생성
    y_data = func(x_data)
    # print('y_data', y_data.shape)

    x_train, x_test = x_data[:n_train], x_data[n_train:]
    y_train, y_test = y_data[:n_train], y_data[n_train:]

#    print(x_train[:5]) # train data 확인
#    print(y_train[:5]) # train data 확인
    
    print('')
    print('func: %s' % func.__name__)
    print('digits: %s ~ %s ' % (min(digits), max(digits)))
    print('x_train: %s' % str(x_train.shape))
    print(x_data[:5])
    print('y_train: %s' % str(y_train.shape))
    print(y_data[:5])
    print('x_test: %s' % str(x_test.shape))
    print('y_test %s' % str(y_test.shape))

    valid_check_interval = 0.5
    bias_value = 0.0
    early_stop_cost = 0.1  # stop learning

    optimizer = tf.train.AdadeltaOptimizer # AdadeltaOptimizer 최저점이 빠르게 출력, 다른 optimizer도 적용해서 검증이 필요 / 하이퍼파라미터가 될만한 것은 다 빼야 함
    activation = tf.sigmoid # 3000 이하 결과 출력에 추천, 만능이 아니므로 다른 것도 검증 필요
    weights_initializer = tf.random_normal_initializer
    n_hiddens = 10 # in one layer
    learning_rate = 0.1 # 발산하다 싶으면 줄여야 함(NaN)
    train_time = 1 # time = 1sec 로 설정하여, cost 감소하는 것만 체크
    
    ### 최적화 테스트 : 시작
    bias_value = 0.0
    activation = tf.nn.relu # 3000 -> 1000으로 감소
    weights_initializer = tf.truncated_normal_initializer # 1000 -> 650으로 감소
    n_hiddens = 1000 # 650 -> 600 감소
    learning_rate = 0.001 # Adeam은 3승 이하 추천 / 590 -> 670으로 증가 | best/total 체크가 필요 : 431/431인 경우로 러닝 시간이 부족한 상황
    train_time = 10 * 60 # best 10분 / cpu에서는 10배 정도 늦음
    ### 최적화 테스트 : 끝

    print('%s -> %s -> %s -> %s -> %s' % (x_train.shape[1], n_hiddens, activation.__name__, n_hiddens, 1))
    print('weights_initializer: %s' % weights_initializer.__name__)
    print('learning_rate: %.4f' % learning_rate)
    print('train_time: %s' % train_time)

    how_many_trains = 3 if train_time <= 1 else 1  # 1초 실행하는 경우, 3번 실험 그 외에는 1번 실험.
    for _ in range(how_many_trains) :
        # time.sleep(1)
        tf.reset_default_graph() # 기존 session을 초기화
        tf.set_random_seed(7942) # tf.random_normal_initializer 사용하기 때문에 설정 필요

        scope_name = '%s.%s' % (func.__name__,DateUtil.current_yyyymmdd_hhmmss()) # graph 겹치지 않게 하기 위해서, func + 날짜 이름으로 설정하는 것을 추천
        x, y, y_hat, cost, rsme, train_step, summary = build_graph(scope_name, n_features, n_hiddens, n_classes, learning_rate, activation=activation, weights_initializer=weights_initializer, bias_value=bias_value)

        try :
            watch = WatchUtil()

            model_file_saved = False
            model_file = os.path.join('%s/workspace/nlp4kor/models/%s_%s/model' % (os.getcwd(), os.path.basename(__name__.replace('.py', '')), func.__name__))
            model_dir = os.path.dirname(model_file)
            # print('model_file: %s' % model_file)
            if not os.path.exists(model_dir):
                # print('model_dir: %s' % model_dir)
                os.makedirs(model_dir)

            config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
            saver = tf.train.Saver() # 최근 5개만 남개 되어서 max_to_keep=None 해야함
            with tf.Session(config=config) as sess:
                sess.run(tf.global_variables_initializer())

                train_writer = tf.summary.FileWriter(TENSORBOARD_LOG_DIR + '/train', sess.graph)
                valid_writer = tf.summary.FileWriter(TENSORBOARD_LOG_DIR + '/valid', sess.graph)

                max_cost = 1e10
                best_epoch, best_cost = 0, 1e10
                watch.start('train')

                running, epoch = True, 0
                stop_timer = TimerUtil(interval_secs=train_time)
                valid_timer = TimerUtil(interval_secs=valid_check_interval)
                stop_timer.start()
                valid_timer.start()
                while running:
                    if stop_timer.is_over():
                        break

                    epoch += 1
                    _, train_rsme, train_summary = sess.run([train_step, rsme, summary], feed_dict={x: x_train, y: y_train})
                    train_writer.add_summary(train_summary, global_step=epoch)
                    train_writer.flush()

                    if valid_timer.is_over():
                        valid_rsme, valid_summary = sess.run([rsme, summary], feed_dict={x: x_test, y: y_test})
                        valid_writer.add_summary(valid_summary, global_step=epoch)
                        valid_writer.flush()
                        if valid_rsme < best_cost:  # cost가 작을 때의 모델만...
                            best_cost = valid_rsme
                            best_epoch = epoch
                            saver.save(sess, model_file)  # 저장하자.
                            model_file_saved = True
                            if epoch % 10 == 0:
                                print('[epoch: %s] rsme (train/valid): %.1f / %.1f model saved' % (epoch, train_rsme, valid_rsme))
                        else:
                            if epoch % 10 == 0:
                                print('[epoch: %s] rsme (train/valid): %.1f / %.1f' % (epoch, train_rsme, valid_rsme))
                        if valid_rsme < early_stop_cost or valid_rsme > max_cost or math.isnan(valid_rsme):  # cost가 너무 크거나, nan이거나 매우 작으면 학습 종료.
                            running = False
                            break
                watch.stop('train')

            if model_file_saved and os.path.exists(model_file + '.index'):
                with tf.Session() as sess:
                    restored = saver.restore(sess, model_file)

                    print('')
                    print('--------TEST----------')
                    watch.start('test')
                    test_rsme, _y_hat = sess.run([rsme, y_hat], feed_dict={x: x_test, y: y_test})

                    print('%s rsme (test): %.1f (epoch best/total: %s/%s), activation: %s, n_hiddens: %s, learning_rate: %s, weights_initializer: %s' % (
                        func.__name__, test_rsme, NumUtil.comma_str(best_epoch), NumUtil.comma_str(epoch), activation.__name__,
                        n_hiddens, learning_rate, weights_initializer.__name__))

                    for i in range(min(5, _y_hat.shape[0])):  # 최대 5개까지만 출력해서 확인
                        print('%s\t->\t%.1f\t(label: %d)' % (x_test[i], _y_hat[i], y_test[i]))
                    print('--------TEST----------')
                    watch.stop('test')
            print(watch.summary())
        except:
            traceback.print_exc()
    print('OK.')