import os
import torch
import time
import ml_collections

## PARAMETERS OF THE MODEL
save_model = True
tensorboard = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_cuda = torch.cuda.is_available()
seed = 666
os.environ['PYTHONHASHSEED'] = str(seed)

n_filts = 16
cosineLR = True # whether use cosineLR or not
n_channels = 3
n_labels = 1





print_frequency = 1
save_frequency = 5000
vis_frequency = 10
epochs = 300
early_stopping_patience = 50


pretrain = False


task_name = 'BUSI_with_GT'
# task_name = 'CHASE'
# task_name = 'Chest Xray'
# task_name = 'CVC-ClinicDB'
# task_name = 'ISIC2017'
# task_name = 'ISIC2018'
# task_name = 'Kvasir-Seg'
# task_name = 'RITE'
# task_name = 'MoNuSeg'
# task_name = 'GlaS'
# task_name = 'DDTI'
# task_name = 'TN3K'
# task_name = 'TG3K'

# used in testing phase, copy the session name in training phase
test_session = "Test_session_07.15_15h46"
# test_session = "Test_session"

model_name = 'MCS_Net'



# learning_rate = 1e-3
learning_rate = 1e-2
# learning_rate = 1e-4

# batch_size = 2
batch_size = 4
# batch_size = 8
# batch_size = 16

# img_size = 128
# img_size = 224
img_size = 256
# img_size = 512

# optimizer = 'AdamW'
# optimizer = 'Adam'
optimizer = 'SGD'

# channel_list = 8,16,24,32,48,64
# channel_list = 8,16,32,48,64,96
# channel_list = 16,24,32,48,64,128
# channel_list = 8,16,32,64,128,160
# channel_list = 16,32,48,64,128,256
# channel_list = 16,32,64,128,160,256
# channel_list = 16,32,64,128,256,512
# depth = 1,1,1,1
# depth = 1,1,2,2
# depth = 1,2,2,4
# depth = 2,2,4,4

train_dataset = './datasets/'+ task_name+ '/Train_Folder/'
val_dataset = './datasets/'+ task_name+ '/Val_Folder/'
test_dataset = './datasets/'+ task_name+ '/Test_Folder/'

session_name       = 'Test_session' + '_' + time.strftime('%m.%d_%Hh%M')

# save_path          ="./data_train_test_session/" + task_name +'/'+ model_name +'/' \
#                     + 'channel_list_' + str(channel_list) + '/' \
#                     + 'lr_'+ str(learning_rate) +' batchsize_'+ str(batch_size) + ' ImgSize_'+ str(img_size) + ' ' + optimizer + '/' \
#                     + session_name + '/'

# save_path          ="./data_train_test_session/" + task_name +'/'+ model_name +'/' \
#                     + 'channel_list_' + str(channel_list) + '/' \
#                     + 'lr_'+ str(learning_rate) +' batchsize_'+ str(batch_size) + ' ImgSize_'+ str(img_size) + optimizer + '/' \
#                     + session_name + '/'

# save_path          ="./data_train_test_session/" + task_name +'/'+ model_name +'/' \
#                     + 'channel_list_' + str(channel_list) + '/' \
#                     + 'lr_'+ str(learning_rate) +'_batchsize_'+ str(batch_size) + '_ImgSize_'+ str(img_size) + '/' \
#                     + session_name + '/'

# save_path          ="./data_train_test_session/" + task_name +'/'+ model_name +'/' \
#                     + 'channel_list_' + str(channel_list) + '/' \
#                     + 'depth_' + str(depth) + '/' \
#                     + session_name + '/'

save_path          ="./data_train_test_session/" + task_name +'/'+ model_name +'/' + session_name + '/'


model_path         = save_path + 'models/'
tensorboard_folder = save_path + 'tensorboard_logs/'
logger_path        = save_path + session_name + ".log"
visualize_path     = save_path + 'visualize_val/'