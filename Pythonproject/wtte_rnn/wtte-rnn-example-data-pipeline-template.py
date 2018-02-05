
# 참조 https://github.com/ragulpr/wtte-rnn-examples/blob/master/examples/data-pipeline-template.ipynb

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import wtte.tte_util as tte
import wtte.transforms as tr

from wtte.pipelines import data_pipeline
from wtte.plots.misc import timeline_aggregate_plot,timeline_plot

pd.options.display.max_rows = 10

df = pd.read_csv('C:/Users/bevis/Downloads/wtte_rnn/tensorflow.csv',error_bad_lines=False,low_memory=False)

print(df.columns.values)
df.fillna(0,inplace=True)

df.rename(columns={"time_sec": "time_int"},inplace=True)

## For transformation df->tensor
id_col='id'
abs_time_col='time_int'

# Put the column indicating 'events' first!
feature_cols= ["n_commits",'files_changed','lines_inserted','lines_deleted']

# feature_cols= ["n_commits"]
constant_cols = []
mean_cols = []

infer_seq_endtime=True
time_sec_interval=60*60*24
timestep_aggregation_dict=dict.fromkeys(feature_cols, "sum")
drop_last_timestep=True

## Create a new sequence-key based on time of first event.
df[id_col] = df.groupby(['author_name'], group_keys=False).apply(lambda g: g.time_int.min().astype(str)+g.author_name.astype(str)).rank(method='dense').astype(int)

df.sort_values([id_col,'time_int'],inplace=True)

df['n_commits'] = 1

plt.scatter(x=pd.to_datetime(df.time_int.values,unit='s'),
            y=df.id,s=0.05, linewidth='0',color='black')
plt.title('datapoints (each dot is a timestamp)')
plt.xlabel('time')
plt.ylabel('user id')
plt.show()

df.head()

import re
# Aggregates by timestep using sum by default. 
# `constant_cols` and `mean_cols` are aggregated using mean
timestep_aggregation_dict = dict.fromkeys(feature_cols, "sum")

for key in timestep_aggregation_dict:
    for query in list(set().union(constant_cols,mean_cols)):
        if re.match(query+'($|_)', key): # binary cols are formatted as `col_level`
            timestep_aggregation_dict[key] = 'mean'
print('how features will be aggregated each day:')
print(timestep_aggregation_dict)

## DataFrame to Tensor
# 1.`timestep_aggregation_dict`를 사용하여 각 열을 모아서 epoch에서 abs (int)`abs_time_col` ex의 해상도를 에포크 데이로 낮추십시오.
# 2. timesteps 사이에 0을 추가하고 'constant_cols'의 값으로 채 웁니다.
# 3. 마지막 '비 사건'이 발생한 곳, 즉 검색어를 작성한 날을 유인합니다. 이것은 검열의 시간입니다.

discrete_time=True
pad_between_steps=True

x, padded_t, seq_ids, df_tmp = data_pipeline(df,
                  id_col=id_col,
                  abs_time_col=abs_time_col,
                  column_names=feature_cols,
                  constant_cols = constant_cols,
                  discrete_time=discrete_time,
                  pad_between_steps=pad_between_steps,
                  infer_seq_endtime=infer_seq_endtime,
                  time_sec_interval=time_sec_interval,
                  timestep_aggregation_dict=timestep_aggregation_dict,
                  drop_last_timestep=drop_last_timestep
                  )
print(x.shape)
df_tmp

## x의 첫 번째 기능은 이벤트 표시기
# We assume the first column is the event-indicator:
events = (np.copy(x[:,:,0])>0).astype(float)
events[np.isnan(x[:,:,0])] = np.nan

assert(np.nansum(df_tmp.n_commits.values>0)==np.nansum(events))

plt.scatter(
              x=pd.to_datetime(df_tmp['time_int'],unit='s').values,
              y=df_tmp['id'].values,
              s=.1,
              linewidth='0', 
              color='black')
plt.title('discretized datapoints in dataframe')
plt.show()

# timeline_aggregate_plot(events,"events in tensor format",cmap="Greys") # 프로그램 종료 됨
plt.show()

## Tensorflow - Feature Engineering - Global Features
# 실시간으로 알려 졌던 모든 타임 스텝에 일별 평균 이벤트를 추가하기 만하면됩니다.
# Add a global feature "mean (number of events) today"
add_global_feature = True
if add_global_feature:
    padded = tr.right_pad_to_left_pad(x[:,:,0])
    x_global = np.nanmean(padded,0)
    x_global = np.expand_dims(x_global,0)
    x_global = padded*0+x_global

    x_global =  tr.left_pad_to_right_pad(x_global)
    x = np.concatenate([x,np.expand_dims(x_global,-1)],-1)
    
    feature_cols.append('mean_commits_global')
    del padded,x_global

## Format Tensor for Training
# tte-values 계산
# test / train 분리 : features / target 조정
# normalize : using training data

def drop_n_last_timesteps(padded,n_timesteps_to_drop,remove_empty_seqs = True):
    # brutal method: simply right align all tensors and simply cut off the last n_timesteps_to_drop
    n_timesteps = x.shape[1]
    padded      = tr.left_pad_to_right_pad(tr.right_pad_to_left_pad(padded)[:,:(n_timesteps-n_timesteps_to_drop)])
    
    if remove_empty_seqs:
        seq_lengths = tr.get_padded_seq_lengths(padded)
        padded = padded[seq_lengths>0]
    return padded

def prep_tensors(x,events):
    # 0. calculate time to event and censoring indicators.
    y  = np.ones([events.shape[0],events.shape[1],2])
    y[:,:,0] = tr.padded_events_to_tte(np.squeeze(events),discrete_time=True)
    y[:,:,1] = tr.padded_events_to_not_censored(np.squeeze(events),discrete_time=True)

    # 1. Disalign features and targets otherwise truth is leaked.
    # 2. drop first timestep (that we now dont have features for)
    # 3. nan-mask the last timestep of features. (that we now don't have targets for)
    events = events[:,1:,]
    y  = y[:,1:]
    x  = np.roll(x, shift=1, axis=1)[:,1:,]
    x  = x + 0*np.expand_dims(events,-1)
    return x,y,events

n_timesteps = x.shape[1]
n_features = x.shape[2]

## Hide 30% of the last timesteps and keep them for testing
frac_timesteps_to_cut = 0.10
n_testing_timesteps = np.floor(n_timesteps*frac_timesteps_to_cut).astype(int)

x_train      = drop_n_last_timesteps(x,n_testing_timesteps)
events_train = drop_n_last_timesteps(events,n_testing_timesteps)

# Do the necessary preparations of the tensors.
x_train,y_train,events_train = prep_tensors(x_train,events_train)
x,y,events    = prep_tensors(x,events)

# Normalize
# transforms normalize_padded float128 windows 오류로 fuction 수정
def normalize_padded(padded, means=None, stds=None):
    """Normalize by last dim of padded with means/stds or calculate them.

        .. TODO::
           * consider importing instead ex:

                from sklearn.preprocessing import StandardScaler, RobustScaler
                robust_scaler = RobustScaler()
                x_train = robust_scaler.fit_transform(x_train)
                x_test  = robust_scaler.transform(x_test)
                ValueError: Found array with dim 3. RobustScaler expected <= 2.

           * Don't normalize binary features
           * If events are sparse then this may lead to huge values.
    """
    # TODO epsilon choice is random
    epsilon = 1e-6
    original_dtype = padded.dtype

    is_flat = len(padded.shape) == 2
    if is_flat:
        padded = np.expand_dims(padded, axis=-1)

    n_features = padded.shape[2]
    n_obs = padded.shape[0] * padded.shape[1]

    if means is None:
        means = np.nanmean(np.longdouble(
            padded.reshape(n_obs, n_features)), axis=0) #np.float128은 windows에서 미제공

    means = means.reshape([1, 1, n_features])
    padded = padded - means

    if stds is None:
        stds = np.nanstd(np.longdouble(
            padded.reshape(n_obs, n_features)), axis=0) #np.float128은 windows에서 미제공

    stds = stds.reshape([1, 1, n_features])
    if (stds < epsilon).any():
        print('warning. Constant cols: ', np.where((stds < epsilon).flatten()))
        stds[stds < epsilon] = 1.0
        # should be (small number)/1.0 as mean is subtracted.
        # Possible prob depending on machine err

    # 128 float cast otherwise
    padded = (padded / stds).astype(original_dtype)

    if is_flat:
        # Return to flat
        padded = np.squeeze(padded)
    return padded, means, stds

x_train, means, stds = normalize_padded(x_train,means=None,stds=None)
x, means, stds = tr.normalize_padded(x,means=means,stds=stds)

seq_lengths = np.count_nonzero(~np.isnan(events), axis=1)
seq_lengths_train = np.count_nonzero(~np.isnan(events_train), axis=1)

# Used for initialization of alpha-bias:
tte_mean_train = np.nanmean(y_train[:,:,0])
mean_u = np.nanmean(y_train[:,:,1])

print('events',events.shape,events.dtype)
print('x min max ',np.nanmin(x),np.nanmax(x))
print('x',x.shape,x.dtype)
print('y',y.shape,y.dtype)
print('x_train',x_train.shape,x_train.dtype)
print('y_train',y_train.shape,y_train.dtype)
print('tte_mean_train: ', tte_mean_train)
print('mean uncensored train: ', np.nanmean(y_train[:,:,1]))

print('x_train size',x_train.nbytes*1e-6,' mb')

## Visualize tensor data
# def timeline_plot(padded,title='',cmap=None,plot=True,fig=None,ax=None):
#     if fig is None or ax is None:
#         fig, ax = plt.subplots(ncols=2, sharey=True,figsize=(12,4))
    
#     ax[0].imshow(padded,interpolation='none', aspect='auto',cmap=cmap,origin='lower')    
#     ax[0].set_ylabel('sequence');
#     ax[0].set_xlabel('sequence time');
        
#     im = ax[1].imshow(tr.right_pad_to_left_pad(padded),interpolation='none', aspect='auto',cmap=cmap,origin='lower')  
#     ax[1].set_ylabel('sequence');
#     ax[1].set_xlabel('absolute time'); #(Assuming sequences end today)
    
#     fig.suptitle(title,fontsize=14)
#     if plot:
#         fig.show()
#         return None,None
#     else:
#         return fig,ax

# def timeline_aggregate_plot(padded,title='',cmap=None,plot=True):
#     fig, ax = plt.subplots(ncols=2,nrows=2,sharex=True, sharey=False,figsize=(12,8))
    
#     fig,ax[0] = timeline_plot(padded,title,cmap=cmap,plot=False,fig=fig,ax=ax[0])
    
#     ax[1,0].plot(np.nanmean(padded,axis=0),lw=0.5,c='black',drawstyle='steps-post')
#     ax[1,0].set_title('mean/timestep')
#     padded = tr.right_pad_to_left_pad(padded)
#     ax[1,1].plot(np.nanmean(padded,axis=0),lw=0.5,c='black',drawstyle='steps-post')
#     ax[1,1].set_title('mean/timestep')

#     fig.suptitle(title,fontsize=14)
#     if plot:
#         fig.show()
#         return None,None
#     else:
#         return fig,ax

# timeline_aggregate_plot(events,"events",cmap="Greys") # 프로그램 종료 됨
# plt.show()

# timeline_aggregate_plot(1-y[:,:,1],"censoring",cmap="Greys") # 프로그램 종료 됨
# plt.show()

print('############## TRAINING SET')
######
# timeline_aggregate_plot(1-y_train[:,:,1],'censoring',cmap='Greys') # 프로그램 종료 됨
# plt.show()
# timeline_aggregate_plot(y_train[:,:,0],'TTE (censored)',cmap='jet') # 프로그램 종료 됨
# plt.show()

train_mask = (False==np.isnan(y_train[:,:,0]))

plt.hist(seq_lengths_train)
plt.title('Distribution of sequence lengths (training set)')
plt.xlabel('sequence length')
plt.show()

plt.hist(y_train[:,:,0][train_mask].flatten(),100)
plt.title('Distribution of censored tte')
plt.ylabel('sequence')
plt.xlabel('t')
plt.show()

plt.hist(y_train[:,:,1][train_mask].flatten(),2)
plt.title('Distribution of censored/non censored points')
plt.xlabel("u")
plt.show()

plt.plot(y_train[0,:,0])
plt.title('example seq tte')
plt.xlabel('t')
plt.show()

print('########## features')
for f in range(x_train.shape[2]):
    try:
        feature_name = feature_cols[f]
    except:
        feature_name = '???'
        
    timeline_aggregate_plot(x_train[:,:,f],feature_name,cmap = 'jet')
    plt.show()
    tmp = x_train[:,:,f].flatten()
    plt.hist(tmp[False==np.isnan(tmp)],10)
    plt.title('Distribution of '+feature_name)
    plt.show()
print('########## ')
del tmp,train_mask

## Masks, weights and validation set
def nanmask_to_keras_mask(x,y,mask_value,tte_mask):
    """nanmask to keras mask.
        :param float mask_value: Use some improbable telltale value 
                                (but not nan-causing)
        :param float tte_mask: something that wont NaN the loss-function
    """
    # Use some improbable telltale value (but not nan-causing)
    x[:,:,:][np.isnan(x)] = mask_value
    y[:,:,0][np.isnan(y[:,:,0])] = tte_mask
    y[:,:,1][np.isnan(y[:,:,1])] = 0.5
    sample_weights = (x[:,:,0]!=mask_value)*1.
    return x,y,sample_weights

mask_value = -1.3371337 

x_train,y_train,sample_weights_train = nanmask_to_keras_mask(x_train,y_train,mask_value,tte_mean_train)

# Pick 10% of the sequences beyond the boundary for validation
valid_indx = np.where(np.random.sample(len(y))<0.1)
x_valid = x[valid_indx].copy()
y_valid = y[valid_indx].copy()
padded = np.copy(y_valid[:,:,0])

x_valid,y_valid,sample_weights_valid = nanmask_to_keras_mask(x_valid,y_valid,mask_value,tte_mean_train)

# Set weights to 0s except the non-nan last timesteps not in training
n_timesteps_to_hide = x_valid.shape[1]-x_train.shape[1]
padded[~np.isnan(padded)] = 1.
padded = tr.right_pad_to_left_pad(padded)
padded[:,:-n_timesteps_to_hide] = padded[:,:-n_timesteps_to_hide]*0
padded = tr.left_pad_to_right_pad(padded)
# timeline_plot(padded,'validation set weights','Greys') # 프로그램 종료 됨

padded[np.isnan(padded)] = 0
sample_weights_valid = np.copy(padded)
plt.show()
del padded,valid_indx

# Initialization value for alpha-bias 
init_alpha = -1.0/np.log(1.0-1.0/(tte_mean_train+1.0) )
init_alpha = init_alpha/mean_u
print('init_alpha: ',init_alpha,'mean uncensored train: ',mean_u)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import wtte.tte_util as tte
import wtte.transforms as tr

from wtte.pipelines import data_pipeline
import wtte.tte_util as tte
import wtte.weibull as weibull
import wtte.wtte as wtte
from wtte.wtte import WeightWatcher

pd.options.display.max_rows = 20
from IPython import display

import keras.backend as K
from keras.models import Sequential

from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Masking
from keras.layers import Lambda
from keras.layers.wrappers import TimeDistributed
from keras.layers.normalization import BatchNormalization

from keras import regularizers
from keras import callbacks
from keras.optimizers import RMSprop,adam
from keras.models import load_model

import wtte.weibull as weibull
import wtte.wtte as wtte
from wtte.wtte import WeightWatcher

# from phased_lstm_keras.PhasedLSTM import PhasedLSTM as PLSTM
np.random.seed(1)
K.set_epsilon(1e-8)

## Callbacks
checkpointer = callbacks.ModelCheckpoint('./tensorboard_log/wtte-rnn/model_checkpoint.h5', 
                          monitor='loss', 
                          verbose=1, 
                          save_best_only=True, 
                          save_weights_only=True, 
                          mode='auto', period=5)

reduce_lr = callbacks.ReduceLROnPlateau(monitor='loss', 
                                        factor  =0.5, 
                                        patience=15, 
                                        verbose=1, 
                                        mode='auto', 
                                        epsilon=0.0001, 
                                        cooldown=0, 
                                        min_lr=1e-8)

nanterminator = callbacks.TerminateOnNaN()
history = callbacks.History()
weightwatcher = WeightWatcher(per_batch =False,per_epoch= True)
n_features = x_train.shape[-1]

## Base model
model = Sequential()
model.add(Masking(mask_value=mask_value,input_shape=(None, n_features)))
model.add(GRU(10,activation='tanh',return_sequences=True,recurrent_dropout=0.1,unroll=False))
model.add(BatchNormalization(axis=-1, momentum=0.9, epsilon=0.01))

model.add(TimeDistributed(Dense(10,activation='tanh')))

## Wtte-RNN part
model.add(TimeDistributed(Dense(2)))
model.add(Lambda(wtte.output_lambda, arguments={"init_alpha":init_alpha, 
                                                "max_beta_value":2.0,
                                                "alpha_kernel_scalefactor":0.5
                                               }))

loss = wtte.loss(kind='discrete',reduce_loss=False).loss_function
model.compile(loss=loss, optimizer=adam(lr=.01,clipvalue=0.5),sample_weight_mode='temporal')
model.summary()

K.set_value(model.optimizer.lr, 0.01)
model.fit(x_train, y_train,
          epochs=300,
          batch_size=300, 
          verbose=1,
          validation_data=(x_valid, y_valid,sample_weights_valid),
          sample_weight = sample_weights_train,
          callbacks=[nanterminator,history,weightwatcher,reduce_lr])

## model 학습이 강제 종료됨 : loss nan
"""
Train on 924 samples, validate on 105 samples
Epoch 1/300

300/924 [========>.....................] - ETA: 3s - loss: 1.7442Batch 1: Invalid loss, terminating training

600/924 [==================>...........] - ETA: 1s - loss: nan   
600/924 [==================>...........] - ETA: 1s - loss: nan<keras.callbacks.History object at 0x00000214ED64CF28>
"""

plt.plot(history.history['loss'],    label='training')
plt.plot(history.history['val_loss'],label='validation')
plt.legend()
plt.show()
weightwatcher.plot()

## Predict
predicted = model.predict(x)
predicted[:,:,1]=predicted[:,:,1]+predicted[:,:,0]*0# lazy re-add of NAN-mask
print(predicted.shape)
print('mean alpha pred',np.nanmean(predicted[:,:,0]))
print('mean beta pred',np.nanmean(predicted[:,:,1]))

# Here you'd stop after transforming to dataframe and piping it back to some database
tr.padded_to_df(predicted,column_names=["alpha","beta"],dtypes=[float,float])

## Scatter
# Pick some random sequence
np.random.seed(12) # 9, 6,5,4 ok
random_selection =np.random.choice(predicted.shape[0], min([5,predicted.shape[0]]))
random_selection = np.sort(random_selection)

# Alpha and beta projections
alpha_flat = predicted[:,:,0][~np.isnan(predicted[:,:,0])].flatten()
beta_flat  = predicted[:,:,1][~np.isnan(predicted[:,:,0])].flatten()

## log-alpha typically makes more sense.

for batch_indx in random_selection:
    from matplotlib.colors import LogNorm
    counts, xedges, yedges, _ = plt.hist2d(alpha_flat, beta_flat, bins=50,norm=LogNorm())
    
    plt.plot(predicted[batch_indx,:,0],predicted[batch_indx,:,1],color='red')
    
    plt.scatter(predicted[batch_indx,0,0],predicted[batch_indx,0,1],
                marker = '*',
                s=50,
                color='red')
    
    plt.title('Predicted params : density')
    plt.xlim([alpha_flat.min(),alpha_flat.max()])
    plt.ylim([beta_flat.min(),beta_flat.max()])
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$\beta$')
    plt.show()

## Individual sequences
drawstyle = 'steps-post'
for batch_indx in random_selection:
    this_seq_len = seq_lengths[batch_indx]
    a = predicted[batch_indx,:this_seq_len,0]
    b = predicted[batch_indx,:this_seq_len,1]
    t = np.array(xrange(len(a)))
    x_this = x[batch_indx,:this_seq_len,:]

    this_tte = y[batch_indx,:this_seq_len,0]
    u = y[batch_indx,:this_seq_len,1]>0
    
    plt.plot(this_tte,label='censored tte',color='black',linestyle='dashed',linewidth=2,drawstyle=drawstyle)
    plt.plot(t[u],this_tte[u],label='uncensored tte',color='black',linestyle='solid',linewidth=2,drawstyle=drawstyle)
    
    plt.plot(weibull.quantiles(a,b,0.75),color='blue',label='pred <0.75',drawstyle=drawstyle)
    plt.plot(weibull.mode(a, b), color='red',linewidth=1,label='pred mode/peak prob',drawstyle=drawstyle)
#    plt.plot(weibull.mean(a, b), color='green',linewidth=1,label='pred mean',drawstyle='steps-post')
    plt.plot(weibull.quantiles(a,b,0.25),color='blue',label='pred <0.25',drawstyle=drawstyle)
    
    plt.xlim(0, this_seq_len)
    plt.ylim(0, min([2*this_tte.max(),2*a.max()]))
    plt.xlabel('time')
    plt.ylabel('time to event')
#     plt.title(authour_names[batch_indx])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

drawstyle = 'steps-post'
for batch_indx in random_selection:
    this_seq_len  = seq_lengths[batch_indx]
    a = predicted[batch_indx,:this_seq_len,0]
    b = predicted[batch_indx,:this_seq_len,1]
    t = np.array(xrange(len(a)))
    x_this = x[batch_indx,:this_seq_len,:]

    ##### Parameters
    # Create axes
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(t, a, color='b')
    ax1.set_xlabel('time')
    ax1.set_ylabel('alpha')

    ax2.plot(t, b, color='r')
    ax2.set_ylabel('beta')

    # Change color of each axis
    def color_y_axis(ax, color):
        """Color your axes."""
        for t in ax.get_yticklabels():
            t.set_color(color)
        return None
    color_y_axis(ax1, 'b')
    color_y_axis(ax2, 'r')
    plt.show()

## Density plot
# Warning: Doesn't look very impressive when we have $\beta$ around $\leq 1$ since pdf is strictly decreasing (hence mode =0).

from wtte.plots.weibull_heatmap import weibull_heatmap
# np.random.seed(1)
for batch_indx in random_selection:
    this_seq_len = seq_lengths[batch_indx]
    if this_seq_len==0:
        next
    a = predicted[batch_indx,:this_seq_len,0]
    b = predicted[batch_indx,:this_seq_len,1]
    t = np.array(xrange(len(a)))
    x_this = x[batch_indx,:this_seq_len,:]

    this_tte = y[batch_indx,:this_seq_len,0]
    censoring_indicator = y[batch_indx,:this_seq_len,1]<1

    fig, ax = plt.subplots(1)

    weibull_heatmap(
        fig,ax,
        t,
        a,
        b,
        max_horizon = int(1.5*this_tte.max()),
        time_to_event=this_tte,
        true_time_to_event=None,
        censoring_indicator = censoring_indicator,
        title='predicted Weibull pmf $p(t,s)$ seq. '+str(batch_indx),
        lw=3.0,
        is_discrete=True,
        resolution=None,
        xax_nbins=10,
        yax_nbins=4 
    )
    plt.show()

## Aggregate view
fig,ax = timeline_plot(y[:,:,0],"tte",plot=False)
plt.show()

padded = weibull.mean(a=predicted[:,:,0],b=predicted[:,:,1])
timeline_aggregate_plot(padded,"predicted (expected value)")
plt.show()

timeline_aggregate_plot(predicted[:,:,0],"alpha")
plt.show()

timeline_aggregate_plot(predicted[:,:,1],"beta")
plt.show()

padded = -weibull.discrete_loglik(a=predicted[:,:,0],b=predicted[:,:,1],t=y[:,:,0],u=y[:,:,1],equality=False)
timeline_aggregate_plot(padded,"logloss")
plt.show()

plt.imshow(y[:,:,0],aspect='auto',interpolation="none",origin='lower')  
plt.title('tte')
plt.colorbar()
plt.show()

padded = weibull.mean(a=predicted[:,:,0],b=predicted[:,:,1])
plt.imshow(padded,aspect='auto',interpolation="none",origin='lower')  
plt.title('expected value')
plt.colorbar()
plt.show()

plt.imshow(predicted[:,:,0],aspect='auto',interpolation="none",origin='lower')  
plt.title('alpha')
plt.colorbar()
plt.show()
plt.imshow(predicted[:,:,1],aspect='auto',interpolation="none",origin='lower')  
plt.title('beta')
plt.colorbar()
plt.show()

## Calibration
# For the uncensored points we assume F(Y) to be uniform.
# Deviations means we have problem of calibration (which we have obviously below.)
# Note this is dependent on sample_weights set to 1 or 0

n_bins = 10
cmf = weibull.cmf(t=y[:,:,0],a=predicted[:,:,0],b=predicted[:,:,1])

cmf = cmf[(~np.isnan(y[:,:,1]))*(y[:,:,1]==1) ]
cmf = cmf[~np.isnan(cmf)]
plt.hist(cmf.flatten(),n_bins,weights = np.ones_like(cmf.flatten())/float(len(cmf.flatten())))
plt.xlabel(r'predicted $F(Y)$')
plt.title('histogram ')
plt.axhline(1.0/n_bins,lw=2,c='red',label='expected')
plt.locator_params(axis='both',nbins=5)
plt.legend()
plt.show()
del cmf

## Evaluated like a sliding box
# Given that we seldom have the truth due to censored data we partly need to rely on the log-loss. Calibration gives us a hint too. A more intuitive feel is to see what the AUC would have been if it was a fixed-window. See 'sliding box' see https://ragulpr.github.io/2016/12/22/WTTE-RNN-Hackless-churn-modeling/#sliding-box-model

# Note this is dependent on sample_weights set to 1 or 0

rom sklearn import metrics
aucs =[]
# compare score to sliding box up to some width, up to last decile
max_box_width = np.sort(seq_lengths)[-len(seq_lengths)//10]

for box_width in xrange(max_box_width):
    if (box_width%10)==0:
        # select only unmasked and comparable datapoints.

        m = ~np.isnan(y[:,:,1])
        # uncensored or within box_width of boundary 
        m[m] = (y[:,:,1][m]==1)|(box_width<y[:,:,1][m]) 

        actual = y[:,:,0][m].flatten()<=box_width
        pred   = weibull.cmf(a=predicted[:,:,0],b=predicted[:,:,1],t=box_width)[m].flatten()

        fpr,tpr,thresholds = metrics.roc_curve(actual,pred)
        auc = metrics.auc(fpr,tpr)
        print('auc: ',auc,' sliding box ',box_width)
        aucs.append(auc)
plt.plot(aucs)
plt.ylabel('AUC')
plt.xlabel('box width')

## Esoteric plots
# Animate predicted churn
# Those with alpha higher than at the their last step is red. Red stream of blood going to the right corner are predicted churners

#### Walk through the timeline and look at the embedding.
# by day
padded = tr.right_pad_to_left_pad(predicted)
events_tmp = tr.right_pad_to_left_pad(events)
# by day since signup
# padded = np.copy(predicted)                            
# events_tmp = np.copy(events)
    
fig, ax = plt.subplots(ncols=2, sharey=False,figsize=(12,4))
cmap = None

ax[1].imshow(events_tmp,interpolation='none', aspect='auto',cmap='Greys',origin='lower')
ax[1].set_title('events');
ax[1].set_ylabel('sequence');
ax[1].set_xlabel('timestep');
ln= ax[1].axvline(x=0,c="red",linewidth=.5,zorder=10)


xlims = [np.nanmin(padded[:,:,0]),np.nanmax(padded[:,:,0])]
ylims = [np.nanmin(padded[:,:,1]),np.nanmax(padded[:,:,1])]
seq_timestep = np.cumsum(np.isnan(padded[:,:,0]),1)

for timestep in xrange(0,predicted.shape[1]):        
    ax[0].cla()

#     from matplotlib.colors import LogNorm
#     m = ~np.isnan(padded[:,timestep,0])
#     ax[0].hist2d(np.log(padded[m,timestep,0]), padded[m,timestep,1], bins=50,norm=LogNorm())
        
    if timestep == 0:
        colors = 'blue'
        this_pred = padded[:,0,:]
    else:
        m = ~np.isnan(padded[:,timestep,0])
        this_pred = padded[m,timestep,:]
    
        alpha_larger = padded[m,timestep-1,0]<padded[m,timestep,0]
        
        # blue : first step
        # Black : same or lower alpha
        # Red : higher alpha
        colors = np.repeat('black',len(alpha_larger))
        colors[alpha_larger] = 'red'
        colors[np.isnan(padded[m,timestep-1,0])] = 'blue'
    
    ax[0].scatter(this_pred[:,0],
                  this_pred[:,1],
                color = colors,
                s=1,
                linewidths=0
               )
    
    ax[0].set_xlim(xlims)
    ax[0].set_ylim(ylims)
    ax[0].set_xscale("log", nonposx='clip')
    
    ax[0].set_title('predicted');
    ax[0].set_xlabel(r'$log(\alpha)$')
    ax[0].set_ylabel(r'$\beta$')

    ln.remove()
    ln= ax[1].axvline(x=timestep,c="red",linewidth=.5,zorder=10)

#     fig_name = './figures/'+format(timestep,'05d')
#     fig.savefig(fig_name,bbox_inches='tight',transparent=False,dpi=100)
#     plt.gcf()
    display.display(plt.gcf())
    display.clear_output(wait=True)

del seq_timestep,xlims,ylims,events_tmp

## 결론
# 관측치의 20 %가 검열되었으며, 이것은 일부 불안정성을 유발합니다.
# 네트워크는 죽음을 인식하는 데 꽤 능숙해진다. (그러므로 그들이 돌아 오지 않을 때 분배를 바꾼다.) 이것은 부분적으로 유물 학습의 한 형태이며, 부분적으로는 좋은 것입니다.

## 구제 수단
# 필자는 Inverse Probability Of Censoring을 기반으로 테스트를 거쳤습니다. 이러한 결과는 첫 번째 장소에서 검열 될 확률에 따라 검열 된 가중 관측치를 가중치로 유지하면서 작동합니다 (동일한 바이너리 네트워크에 과부하하여 사용).
# 긴 시퀀스에는 타임 스텝이 더 많으므로 가중치 적용 클리핑을 사용하지 않으면 손실에 더 많은 영향을 미칩니다.
# 부분적으로 만 작동하고 검열 된 데이터 포인트에 대한 높은 예측 불이익을 초래할 수있는 덕트 테이프 솔루션. 즉, F (Y)> 일정한 임계 값 및 Y> 일정한 임계 값이있을 때마다 F (Y) 패널티를 추가합니다.
# 구제 수단을 사용하지 마십시오. 일부 관측 값이 무한대로 올라갈 것으로 예상된다는 사실이 요구 될 수 있습니다.