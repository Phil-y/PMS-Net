import torch.optim
from Load_Dataset import ValGenerator, ImageToImage2D
from torch.utils.data import DataLoader
import warnings

warnings.filterwarnings("ignore")
import Config as config
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from datetime import datetime



from networks.PMS_Net import PMS_Net




from Utils import *
import cv2

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--num_classes', type=int,
                    default=1, help='output channel of network')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()
def show_image_with_dice(predict_save, labs, save_path):

    tmp_lbl = (labs).astype(np.float32)
    tmp_3dunet = (predict_save).astype(np.float32)
    dice_pred = 2 * np.sum(tmp_lbl * tmp_3dunet) / (np.sum(tmp_lbl) + np.sum(tmp_3dunet) + 1e-5)
    # dice_show = "%.3f" % (dice_pred)
    iou_pred = jaccard_score(tmp_lbl.reshape(-1),tmp_3dunet.reshape(-1))
    # fig, ax = plt.subplots()
    # plt.gca().add_patch(patches.Rectangle(xy=(4, 4),width=120,height=20,color="white",linewidth=1))
    if config.task_name == "MoNuSeg":
        predict_save = cv2.pyrUp(predict_save,(448,448))
        predict_save = cv2.resize(predict_save,(224,224))
        # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #定义一个核
        # predict_save = cv2.filter2D(predict_save, -1, kernel=kernel)
        cv2.imwrite(save_path,predict_save * 255)
    else:
        cv2.imwrite(save_path,predict_save * 255)
    # plt.imshow(predict_save * 255,cmap='gray')
    # plt.text(x=10, y=24, s="Dice:" + str(dice_show), fontsize=5)
    # plt.axis("off")
    # #remove the white borders
    # height, width = predict_save.shape
    # fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    # plt.margins(0, 0)
    # plt.savefig(save_path, dpi=2000)
    # plt.close()
    return dice_pred, iou_pred

def vis_and_save_heatmap(model, input_img, img_RGB, labs, vis_save_path, dice_pred, dice_ens):
    model.eval()

    output = model(input_img.cuda())
    pred_class = torch.where(output>0.5,torch.ones_like(output),torch.zeros_like(output))
    predict_save = pred_class[0].cpu().data.numpy()
    predict_save = np.reshape(predict_save, (config.img_size, config.img_size))
    dice_pred_tmp, iou_tmp = show_image_with_dice(predict_save, labs, save_path=vis_save_path+'_predict'+model_type+'.jpg')



    input_img.to('cpu')

    # input_img = input_img[0].transpose(0, -1).cpu().detach().numpy()
    # labs = labs[0]
    # output = output[0, 0, :, :].cpu().detach().numpy()

    # if (True):
    #     pickle.dump({
    #         'input': input_img,
    #         'output': (output >= 0.5) * 1.0,
    #         'ground_truth': labs,
    #         'dice': dice_pred_tmp,
    #         'iou': iou_tmp
    #     },
    #         open(vis_save_path + '.pkl', 'wb'))

    # if (True):
    #     plt.figure(figsize=(10, 3.3))
    #     plt.subplot(1, 3, 1)
    #     plt.imshow(input_img)
    #     plt.subplot(1, 3, 2)
    #     plt.imshow(labs, cmap='gray')
    #     plt.subplot(1, 3, 3)
    #     plt.imshow((output >= 0.5) * 1.0, cmap='gray')
    #     plt.suptitle(f'Dice score : {np.round(dice_pred_tmp, 3)}\nIoU : {np.round(iou_tmp, 3)}')
    #     plt.tight_layout()
    #     plt.savefig(vis_save_path)
    #     plt.close()


    return dice_pred_tmp, iou_tmp



if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    test_session = config.test_session
    if config.task_name == "BUSI_with_GT":
        test_num = 85
        model_type = config.model_name
        model_path = "./data_train_test_session/BUSI_with_GT/"+model_type + "/" +test_session+"/models/best_model-"+model_type+".pth.tar"

    elif config.task_name == "CHASE":
        test_num = 52
        model_type = config.model_name
        model_path = "./data_train_test_session/CHASE/"+model_type+"/"   + test_session+"/models/best_model-"+model_type+".pth.tar"

    elif config.task_name == "Chest Xray":
        test_num = 144
        model_type = config.model_name
        model_path = "./data_train_test_session/Chest Xray/"+model_type+"/" + test_session+"/models/best_model-"+model_type+".pth.tar"

    elif config.task_name == "CVC-ClinicDB":
        test_num = 122
        model_type = config.model_name
        model_path = "./data_train_test_session/CVC-ClinicDB/" + model_type + "/"  + test_session + "/models/best_model-" + model_type + ".pth.tar"

    elif config.task_name == "DDTI":
        test_num = 126
        model_type = config.model_name
        model_path = "./data_train_test_session/DDTI/" + model_type + "/"  + test_session + "/models/best_model-" + model_type + ".pth.tar"
        # model_path = "./data_train_test_session/DDTI/" + model_type + "/" + path + test_session + "/models/best_model-" + model_type + ".pth.tar"

    elif config.task_name == "ISIC2017":
        test_num = 433
        model_type = config.model_name
        model_path = "./data_train_test_session/ISIC2017/" + model_type + "/"  + test_session + "/models/best_model-" + model_type + ".pth.tar"

    elif config.task_name == "ISIC2018":
        test_num = 535
        model_type = config.model_name
        model_path = "./data_train_test_session/ISIC2018/" + model_type + "/"  + test_session + "/models/best_model-" + model_type + ".pth.tar"

    elif config.task_name == "Kvasir-Seg":
        test_num = 200
        model_type = config.model_name
        model_path = "./data_train_test_session/Kvasir-Seg/" + model_type + "/"  + test_session + "/models/best_model-" + model_type + ".pth.tar"

    elif config.task_name == "MoNuSeg":
        test_num = 14
        model_type = config.model_name
        model_path = "./data_train_test_session/MoNuSeg/"+model_type+"/" +test_session+"/models/best_model-"+model_type+".pth.tar"

    elif config.task_name == "TG3K":
        test_num = 708
        model_type = config.model_name
        model_path = "./data_train_test_session/TG3K/"+model_type+"/" +test_session+"/models/best_model-"+model_type+".pth.tar"

    elif config.task_name == "TN3K":
        test_num = 614
        model_type = config.model_name
        model_path = "./data_train_test_session/TN3K/"+model_type+"/" +test_session+"/models/best_model-"+model_type+".pth.tar"

    elif config.task_name == "GlaS":
        test_num = 35
        model_type = config.model_name
        model_path = "./data_train_test_session/GlaS/"+model_type+"/" +test_session+"/models/best_model-"+model_type+".pth.tar"

    elif config.task_name == "RITE":
        test_num = 8
        model_type = config.model_name
        model_path = "./data_train_test_session/RITE/"+model_type+"/" +test_session+"/models/best_model-"+model_type+".pth.tar"

    save_path  = "./data_train_test_session/" + config.task_name +'/'+ model_type +'/'  + test_session + '/'
    vis_path = "./data_train_test_session/" + config.task_name + '_visualize_test/'
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)

    checkpoint = torch.load(model_path, map_location='cuda')

    fp = open(save_path + 'test_result', 'a')
    fp.write(str(datetime.now()) + '\n')

    # if model_type == 'ACC_UNet':
    #     model = ACC_UNet()

    if model_type == 'MCS_Net':
        model = MCS_Net()




    else: raise TypeError('Please enter a valid name for the model type')

    model = model.cuda()
    if torch.cuda.device_count() > 1:
        print ("Let's use {0} GPUs!".format(torch.cuda.device_count()))
        model = nn.DataParallel(model, device_ids=[0,1,2,3])
    model.load_state_dict(checkpoint['state_dict'])
    print('Model loaded !')
    tf_test = ValGenerator(output_size=[config.img_size, config.img_size])
    test_dataset = ImageToImage2D(config.test_dataset, tf_test,image_size=config.img_size)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    dice_pred = 0.0
    iou_pred = 0.0
    dice_ens = 0.0
    acc_pred = 0.0


    import time
    from metrics import Metrics, evaluate

    all_start = time.time()
    metrics = Metrics(['precision', 'Sensitivity', 'specificity', 'accuracy', 'IOUJaccard', 'dice_or_f1', 'MAEMAEMAE', 'auc_roc','f1_score'])
    total_iou = 0
    total_cost_time = 0


    with tqdm(total=test_num, desc='Test visualize', unit='img', ncols=70, leave=True) as pbar:
        for i, (sampled_batch, names) in enumerate(test_loader, 1):
            test_data, test_label = sampled_batch['image'], sampled_batch['label']

            arr=test_data.numpy()
            arr = arr.astype(np.float32())
            lab=test_label.data.numpy()
            img_lab = np.reshape(lab, (lab.shape[1], lab.shape[2])) * 255
            fig, ax = plt.subplots()
            plt.imshow(img_lab, cmap='gray')
            plt.axis("off")
            height, width = config.img_size, config.img_size
            fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.savefig(vis_path+str(i)+"_lab.jpg", dpi=300)
            plt.close()



            test_data = test_data.cuda()
            test_label = test_label.cuda()

            start = time.time()
            pred = model.forward(test_data)
            cost_time = time.time() - start


            Precision, Recall, Specificity, accuracy, IoU, DICE, MAE, auc_roc, f1_score= evaluate(pred, test_label)
            metrics.update(precision=Precision, Sensitivity=Recall, specificity=Specificity, accuracy=accuracy, IOUJaccard=IoU,
                           dice_or_f1=DICE,  MAEMAEMAE=MAE,  auc_roc=auc_roc, f1_score=f1_score)


            # total_iou += iou
            # total_cost_time += cost_time
            print(config.model_name)
            metrics_result = metrics.mean(test_num)
            metrics_result['inference_time'] = time.time() - all_start
            print("Test Result:")
            print(
                'precision: %.4f, recall_Sensitivity: %.4f, specificity: %.4f, accuracy: %.4f, iou: %.4f, dice: %.4f, mae: %.4f, auc: %.4f, f1_score: %.4f, inference_time: %.4f'
                % (metrics_result['precision'], metrics_result['Sensitivity'], metrics_result['specificity'],
                   metrics_result['accuracy'], metrics_result['IOUJaccard'], metrics_result['dice_or_f1'],
                     metrics_result['MAEMAEMAE'], metrics_result['auc_roc'], metrics_result['f1_score'],metrics_result['inference_time'])
                  )

            # print("total_cost_time:", total_cost_time)
            # print("loop_cost_time:", time.time() - all_start)


            # evaluation_dir = os.path.sep.join([save_path, 'metrics',  '/'])
            # if not os.path.exists(evaluation_dir):
            #     os.makedirs(evaluation_dir)



            keys_txt = ''
            values_txt = ''
            for k, v in metrics_result.items():
                if k != 'mae' and k != 'hd' and k != 'inference_time':
                    v = 100 * v

                # keys_txt += k + '\t'

                keys_txt  +='   ' + k + '\t'
                values_txt += '    %.2f' % v + '\t'+ '\t'

            name = keys_txt + '\n'
            text = values_txt + '\n'
            metrics_path= save_path + '/' + config.model_name + '_metrics' + '.txt'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            with open(metrics_path, 'a+') as f:
                f.write(name)
                f.write(text)
            print(f'metrics saved in {metrics_path}')
            print("------------------------------------------------------------------")







            values_txt = ' '
            for k, v in metrics_result.items():
                if k != 'mae' and k != 'hd' and k != 'inference_time':
                    v = 100 * v

                # keys_txt += '   ' + k + '\t'
                values_txt += '%.2f' % v + '\t'

            # name = keys_txt + '\n'
            text = values_txt + '\n'
            metrics_path = save_path + '/' + config.model_name + '_metrics_draw' + '.txt'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            with open(metrics_path, 'a+') as f:
                # f.write(name)
                f.write(text)
            print(f'metrics saved in {metrics_path}')
            print("------------------------------------------------------------------")



            input_img = torch.from_numpy(arr)
            dice_pred_t,iou_pred_t = vis_and_save_heatmap(model, input_img, None, lab,
                                                          vis_path+str(i),
                                               dice_pred=dice_pred, dice_ens=dice_ens)
            dice_pred+=dice_pred_t
            iou_pred+=iou_pred_t
            torch.cuda.empty_cache()
            pbar.update()
    print ("dice_pred",dice_pred/test_num)
    print ("iou_pred",iou_pred/test_num)

    fp.write(f"dice_pred : {dice_pred/test_num}\n")
    fp.write(f"iou_pred : {iou_pred/test_num}\n")
    fp.close()



