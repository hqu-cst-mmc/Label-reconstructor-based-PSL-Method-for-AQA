import argparse
import logging
import os
from os import listdir
from os.path import isfile, join, isdir

import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import Dataset

import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils

# import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
# import cv2
from torchvision.transforms import ToPILImage
from torch.optim.lr_scheduler import StepLR
from p3d_model import P3D199
from total_score import TTC
from feature_concat import TFN1,TFN2,TFN3,TFN4,TFN5
# from label_encode import LBE
# from i3dpt import Unit3Dpy, I3D
from utils import transfer_model
from dataset import divingDataset
# from visualize import make_dot
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

np.seterr(divide='ignore', invalid='ignore')


# def setup_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True
#



logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
parser = argparse.ArgumentParser(description="Diving")

parser.add_argument("--load", default=0, type=int,
                    help="Load saved network weights. 0 represent don't load; other number represent the model number")
parser.add_argument("--save", default=0, type=int,
                    help="Save network weights. 0 represent don't save; number represent model number")
parser.add_argument("--epochs", default=150, type=int,
                    help="Epochs through the data. (default=65)")
parser.add_argument("--learning_rate", "-lr", default=0.0001, type=float,
                    help="Learning rate of the optimization. (default=0.0001)")
parser.add_argument("--batch_size", default=8, type=int,
                    help="Batch size for training. (default=16)")
parser.add_argument("--optimizer", default="Adam", choices=["SGD", "Adadelta", "Adam"],
                    help="Optimizer of choice for training. (default=Adam)")
parser.add_argument("--gpuid", default=[], nargs='+', type=str,
                    help="ID of gpu device to use. Empty implies cpu usage.")
parser.add_argument("--size", default=160, type=int,
                    help="size of images.")
parser.add_argument("--task", default='score', type=str,
                    help="task to be overall score or the difficulity level")
parser.add_argument("--only_last_layer", default=0, type=int,
                    help="whether choose to freezen the parameters for all the layers except the linear layer on the pre-trained model")
parser.add_argument("--normalize", default=1, type=int,
                    help="do the normalize for the images")
parser.add_argument("--lr_steps", default=[15, 60], type=int, nargs="+",
                    help="steps to decay learning rate")
parser.add_argument("--use_trained_model", default=1, type=int,
                    help="whether use the pre-trained model on kinetics or not")
parser.add_argument("--random", default=0, type=int,
                    help="random sapmling in training")
parser.add_argument("--test", default=0, type=int,
                    help="whether get into the whole test mode (not recommend) ")
parser.add_argument("--stop", default=0.99, type=float,
                    help="Perform early stop")
parser.add_argument("--tcn_range", default=[1, 2, 3, 4, 5], type=list,
                    help="which part of tcn to use (0 is not using)")
parser.add_argument("--downsample", default=2, type=int,
                    help="downsample rate for stages")
parser.add_argument("--region", default=0, type=int,
                    help="1 or 2. 1 is stage 0, 1, 2, 3 (without sending); 2 is stage 0, 1, 2 (without entering into water and ending)")
parser.add_argument("--allstage", default=1, type=int,
                    help="sampled cover all stage")


def main(options):
    # Path to the directories of features and labels
    # train_file = './data_files/new_split/new_split/training_idx1.npy'
    # test_file = './data_files/new_split/new_split/testing_idx1.npy'

    train_file = './data_files/training_idx.npy'
    test_file = './data_files/testing_idx.npy'
    data_folder = '/home/donglijia/dlj/diving-score-master/frames'
    diff_file = './data_files/difficulty_level.npy'

    range_file = './data_files/tcn_time_point.npy'
    if options.task == "score":
        label_file = './data_files/overall_scores.npy'
        exe_file = './data_files/execution_score.npy'
        encoder_file = './data_files/norm_exescore.npy'
    else:
        label_file = './data_files/difficulty_level.npy'

    if options.normalize:
        transformations = transforms.Compose([transforms.Scale((options.size, options.size)),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                              ])
    else:
        transformations = transforms.Compose([transforms.Scale((options.size, options.size)),
                                              transforms.ToTensor()
                                              ])
    # 这步加载数据的
    #     if options.allstage:
    #         for stage in range(1,5):
    dset_train = divingDataset(data_folder, train_file, label_file, diff_file, exe_file, encoder_file,range_file, transformations,
                               tcn_range=options.tcn_range, random=options.random, size=options.size,
                               downsample=options.downsample, region=options.region, allstage=options.allstage)
    # else:
    #     dset_train = divingDataset(data_folder, train_file, label_file, range_file, transformations,
    #                            tcn_range=options.tcn_range, random=options.random, size=options.size,
    #                            downsample=options.downsample, region=options.region, allstage=options.allstage)

    if options.test:
        # print 'test in train'
        dset_test = divingDataset(data_folder, test_file, label_file, diff_file, exe_file, encoder_file,range_file, transformations, test=1,
                                  tcn_range=options.tcn_range,
                                  size=options.size)
        options.batch_size = 10
    else:
        # print 'no test in train'
        writer1 = SummaryWriter('runs/exp/train-ag')
        writer2 = SummaryWriter('runs/exp/test-ag')

        dset_test = divingDataset(data_folder, test_file, label_file, diff_file, exe_file,encoder_file, range_file, transformations,
                                  tcn_range=options.tcn_range, random=options.random, test=0, size=options.size,
                                  downsample=options.downsample, region=options.region, allstage=options.allstage)

    train_loader = DataLoader(dset_train,
                              batch_size=options.batch_size,
                              shuffle=True,
                              )

    test_loader = DataLoader(dset_test,
                             # batch_size=int(options.batch_size/2),
                             batch_size=options.batch_size,
                             shuffle=True,
                             )

    # use_cuda=1
    use_cuda = (len(options.gpuid) >= 1)
    # if options.gpuid:
    # cuda.set_device(int(options.gpuid[0]))

    # Initial the model
    if options.use_trained_model:
        model = P3D199(pretrained=True, num_classes=400)
        # for name, value in model.named_parameters():
        #     if name == 'fc':
        #         value.requires_grad = False
    else:
        model = P3D199(pretrained=False, num_classes=400)

    for param in model.parameters():
        param.requires_grad = True

    if options.only_last_layer:
        for param in model.parameters():
            param.requires_grad = False

    model = transfer_model(model, num_classes=1, model_type="P3D")
    ttc_model = TTC().cuda()
    tfc_model1 = TFN1().cuda()
    tfc_model2 = TFN2().cuda()
    tfc_model3 = TFN3().cuda()
    tfc_model4 = TFN4().cuda()
    tfc_model5 = TFN5().cuda()
    # label_encode_model = LBE().cuda()

    if use_cuda:
        model.cuda()
    #	model = nn.DataParallel(model, devices=gpuid)

    start_epoch = 0
    if options.load:
        logging.info("=> loading checkpoint" + str(options.load) + ".tar")
        checkpoint = torch.load('./models/for_layers_ablation(1FC)/model_for_sub' + str(options.load) + '.tar')
        # checkpoint = torch.load('./results/new_split/checkpoint' + str(options.load) + '.tar')

        start_epoch = checkpoint['epoch']
        logging.info("=> start epoch: " + str(start_epoch))
        model.load_state_dict(checkpoint['state_dict1'])
        ttc_model.load_state_dict(checkpoint['state_dict2'])
        tfc_model1.load_state_dict(checkpoint['state_dict3'])
        tfc_model2.load_state_dict(checkpoint['state_dict4'])
        tfc_model3.load_state_dict(checkpoint['state_dict5'])
        tfc_model4.load_state_dict(checkpoint['state_dict6'])
        tfc_model5.load_state_dict(checkpoint['state_dict7'])



    # criterion = nn.MSELoss()
    criterion = nn.MSELoss()
    # fc_for_total_score = nn.Linear(5, 1).cuda()

    if options.only_last_layer:
        optimizer = eval("torch.optim." + options.optimizer)(model.fc.parameters(), lr=options.learning_rate,weight_decay=0.01)
    else:
        if options.optimizer == "SGD":
            optimizer = torch.optim.SGD([{'params':[param for name, param in model.named_parameters()
                                                                         if 'fc' not in name]},{'params':ttc_model.parameters()}],
                                        options.learning_rate,
                                        momentum=0.9,
                                        weight_decay=5e-4)
        else:
            optimizer = eval("torch.optim." + options.optimizer)([{'params':model.parameters()},{'params':ttc_model.parameters()},
                                                                  {'params':tfc_model1.parameters()},{'params':tfc_model2.parameters()}
                                                                  ,{'params':tfc_model3.parameters()}
                                                                  ,{'params':tfc_model4.parameters()},{'params':tfc_model5.parameters()},
                                                                  ], lr=options.learning_rate, weight_decay=5e-4)

    scheduler = StepLR(optimizer, step_size=options.lr_steps[0], gamma=0.1)
    all_test_loss =[]
    if not options.test:
        all_train_loss = []
        if options.allstage:
            for epoch_i in range(0, options.epochs):
                logging.info("At {0}-th epoch.".format(epoch_i))
                # if (epoch_i == 30):
                #     print("At epoch 30:",all_train_output)
                #     print(all_labels)
                train_loss = 0.0
                all_train_output = []
                all_labels = []
                all_exescores = []
                all_exes_output = []

                for it, train_data in enumerate(train_loader, 0):
                    loss_sum = 0
                    # 每个it里8个batch
                    vid_tensor_save =[]
                    train_output_sum = []
                    vid_tensor_sum = []
                    all_score = 0
                    train_output_sum = 0
                    vid_tensor_save = []
                    for tcncover in range(1, 6):
                        vid_tensor, labels, diffs, exes,encs  = train_data
                        if use_cuda:
                            vid_tensor_single, labels_single, diffs_single, exe_single, encoder_single = Variable(
                                vid_tensor[tcncover - 1]).cuda(), Variable(
                                labels[tcncover - 1]).cuda(), Variable(diffs[tcncover - 1]).cuda(), Variable(exes[tcncover - 1]).cuda(), Variable(encs[tcncover - 1]).cuda()
                            labels_single = labels_single[:,np.newaxis]
                            diffs_single = diffs_single[:,np.newaxis]
                            exe_single = exe_single[:,np.newaxis]
                            encoder_single = encoder_single[:,np.newaxis]

                        else:
                            vid_tensor, labels = Variable(vid_tensor), Variable(labels)
                        model.train()
                        train_output = model(vid_tensor_single)
                        ## [batch,2048]
                        # train_output = train_output[1]
                        tfc_model1.train()
                        tfc_model2.train()
                        tfc_model3.train()
                        tfc_model4.train()
                        tfc_model5.train()
                        # 执行分数归一化 拼接到2048后面
                        # exe_fea = torch.sigmoid((labels_single - mean)/std)
                        train_output = torch.cat((train_output, encoder_single), 1)
                        tfc_model = eval('tfc_model' + str(tcncover))
                        train_output = tfc_model(train_output)
                        vid_tensor_save.append(train_output)
                    ## [2,5]
                    train_output_sum = torch.cat((vid_tensor_save[0],vid_tensor_save[1],
                                                   vid_tensor_save[2],vid_tensor_save[3],vid_tensor_save[4]),1)
                    ttc_model.train()
                    ## [2,1]
                    ## 结果是归一化后的分数
                    exe_score_output = ttc_model(train_output_sum)* 30
                    train_output_sum = exe_score_output * diffs_single

                    loss = criterion(train_output_sum, labels_single)
                    weights_tscore = ttc_model.fc_ttc.weight.data.cpu().numpy()
                    ##存放整个视频 每个阶段的特征
                    # loss2 = criterion(train_output_sum, labels_single)
                    # loss = loss1
                    all_train_output = np.append(all_train_output, train_output_sum.data.cpu().numpy()[:, 0])
                    all_labels = np.append(all_labels, labels_single.data.cpu().numpy())
                    all_exes_output = np.append(all_exes_output, exe_score_output.data.cpu().numpy()[:, 0])
                    all_exescores = np.append(all_exescores,exe_single.data.cpu().numpy())
                    if it % 100 == 0:
                    # # print(train_output.data.cpu().numpy()[0][0], '-', labels_single.data.cpu().numpy()[0])
                    #     # logging.info("loss at iteration {0}: {1}".format(it, loss.item()))
                        f = open('./results/layer_ablation(1FC)/result_ours_train.txt', mode='a')
                        f.write('\n' + str(train_output_sum.data.cpu().numpy()[0][0]) + '-' + str(labels_single.data.cpu().numpy()[0]))
                        f.write('\n' + str(exe_score_output.data.cpu().numpy()[0][0]) + '-' + str(
                        exe_single.data.cpu().numpy()[0]))
                        f.write('\n')
                    train_loss = train_loss + loss.item()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()



                scheduler.step()
                    #Variable转成numpy
                train_avg_loss = (train_loss / (len(dset_train) / options.batch_size))

                rho1, p_val1 = spearmanr(all_train_output, all_labels)
                rho2, p_val2 = spearmanr(all_exes_output, all_exescores)
                n = len(all_labels)
                mse1 = sum(np.square(all_labels - all_train_output)) / n
                mse2 = sum(np.square(all_exescores - all_exes_output)) / n

                logging.info(
                    "Average training loss value per instance is {0}, the overall-score corr is {1} , the execution scores corr is {2},the overall-score mse is {3} , the execution scores mse is {4} at the end of epoch {5}".format(
                        train_avg_loss, rho1, rho2,mse1,mse2, epoch_i))
                all_train_loss = np.append(all_train_loss, train_avg_loss)
                f = open('./results/layer_ablation(1FC)/train—loss.txt', mode='a')
                f.write('\n' + 'train loss at epoch' + str(epoch_i) + ':' + '\n')
                f.write(str(train_avg_loss))
                f.write('\n')
                writer1.add_scalar('dlosg-model_for_sub_value', train_avg_loss, epoch_i + 1)



                if options.save:
                    torch.save({
                        'epoch': epoch_i + 1,
                        'state_dict1': model.state_dict(),
                        'state_dict2':ttc_model.state_dict(),
                        'state_dict3':tfc_model1.state_dict(),
                        'state_dict4': tfc_model2.state_dict(),
                        'state_dict5': tfc_model3.state_dict(),
                        'state_dict6': tfc_model4.state_dict(),
                        'state_dict7': tfc_model5.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }, './models/for_layers_ablation(1FC)/model_for_sub' + str(options.save) + '.tar')
                # if rho1 > options.stop:
                #     break
                # # main test loop
                with torch.no_grad():
                    # setup_seed(20)

                    test_loss = 0.0
                    all_test_output = []
                    all_labels = []
                    all_exescores = []
                    all_exes_output = []
                    for it, test_data in enumerate(test_loader, 0):
                        test_output_sum = 0
                        vid_tensor_sum = 0
                        all_score = 0
                        vid_tensor_save = []
                        for tcncover in range(1, 6):
                            vid_tensor, labels, diffs, exes, encs = test_data
                            if use_cuda:
                                vid_tensor_single, labels_single, diffs_single, exe_single, encoder_single = Variable(
                                    vid_tensor[tcncover - 1]).cuda(), Variable(
                                    labels[tcncover - 1]).cuda(), Variable(
                                    diffs[tcncover - 1]).cuda(), Variable(exes[tcncover - 1]).cuda(), Variable(
                                    encs[tcncover - 1]).cuda()
                                labels_single = labels_single[:, np.newaxis]
                                diffs_single = diffs_single[:, np.newaxis]
                                exe_single = exe_single[:, np.newaxis]
                                encoder_single = encoder_single[:, np.newaxis]
                            else:
                                vid_tensor, labels = Variable(vid_tensor), Variable(labels)

                            model.eval()

                            test_output = model(vid_tensor_single)

                            test_output = torch.cat((test_output, encoder_single), 1)
                            ## [batch,1]
                            tfc_model1.eval()
                            tfc_model2.eval()
                            tfc_model3.eval()
                            tfc_model4.eval()
                            tfc_model5.eval()
                            tfc_model = eval('tfc_model' + str(tcncover))
                            test_output = tfc_model(test_output)
                            vid_tensor_save.append(test_output)
                            ## [2,5]
                        test_output_sum = torch.cat((vid_tensor_save[0], vid_tensor_save[1],
                                                      vid_tensor_save[2], vid_tensor_save[3], vid_tensor_save[4]), 1)

                        ttc_model.eval()

                        exe_score_output = ttc_model(test_output_sum) * 30
                        test_output_sum = exe_score_output * diffs_single
                        loss = criterion(test_output_sum, labels_single)

                        all_exes_output = np.append(all_exes_output, exe_score_output.data.cpu().numpy()[:, 0])
                        all_exescores = np.append(all_exescores, exe_single.data.cpu().numpy())

                        all_test_output = np.append(all_test_output, test_output_sum.data.cpu().numpy()[:, 0])
                        all_labels = np.append(all_labels, labels_single.data.cpu().numpy())
                        if it % 10 == 0:
                            # # print(train_output.data.cpu().numpy()[0][0], '-', labels_single.data.cpu().numpy()[0])
                            #     # logging.info("loss at iteration {0}: {1}".format(it, loss.item()))
                            f = open('./results/layer_ablation(1FC)/result_ours_test.txt', mode='a')
                            f.write('\n' + str(test_output_sum.data.cpu().numpy()[0][0]) + '-' + str(
                                labels_single.data.cpu().numpy()[0]))
                            f.write('\n')
                        test_loss += loss.item()

                    test_avg_loss = test_loss / (len(dset_test) / options.batch_size)
                    rho1, p_val1 = spearmanr(all_test_output, all_labels)
                    rho2, p_val2 = spearmanr(all_exes_output, all_exescores)
                    n = len(all_labels)
                    mse1 = sum(np.square(all_labels - all_test_output)) / n
                    mse2 = sum(np.square(all_exescores - all_exes_output)) / n

                    logging.info(
                        "Average testing loss value per instance is {0}, the overall-score corr is {1} , the execution scores corr is {2},the overall-score mse is {3} , the execution scores mse is {4} at the end of epoch {5}".format(
                            test_avg_loss, rho1, rho2, mse1, mse2, epoch_i))
                    # test_corr = test_corr.append(rho)
                    # print('test_corr=',test_corr)
                    all_test_loss = np.append(all_test_loss, test_avg_loss)
                    writer2.add_scalar('dlosg-model_for_sub_value', test_avg_loss, epoch_i + 1)

                if rho1 > options.stop:
                    break

        # f = open('weights_of_5scores.txt', mode='a')
        # f.write('weights of 5scores:' + str(weights_tscore[0]))
        # f.write('\n')
        #######################################################################################################################
        ### loss可视化
        epoch_range = []
        for i in range(0, epoch_i + 1):
            epoch_range.append(i)
        plt.figure()
        plt.plot(epoch_range, all_train_loss, 'b')
        plt.plot(epoch_range, all_test_loss, 'g')
        plt.show()
    else:
        # the last test for visualization
      with torch.no_grad():

          # setup_seed(20)

          weights_tscore = ttc_model.fc_ttc.weight.data.cpu().numpy()
          f = open('./results/weights_of_5scores.txt', mode='a')
          f.write('weights of 5scores:' + str(weights_tscore[0]))
          f.write('\n')

          test_loss = 0.0
          all_test_output = []
          all_labels = []
          all_exes_output = []
          all_exescores = []

          ### old split:300/70
          all_test_samples_subscores = [[] for i in range(70)]
          all_training_samples_subscores = [[] for i in range(300)]

          ### new split:288/72
          # all_test_samples_subscores = [[] for i in range(72)]
          # all_training_samples_subscores = [[] for i in range(298)]
          start= -options.batch_size

          for it, test_data in enumerate(train_loader, 0):
          # for it, test_data in enumerate(test_loader, 0):
                test_output_sum = 0
                vid_tensor_sum = 0
                all_score = 0
                start += options.batch_size
                vid_tensor_save = []
                for tcncover in range(1, 6):
                    vid_tensor, labels, diffs, exes, encs , video_name = test_data
                    if use_cuda:
                        vid_tensor_single, labels_single, diffs_single, exe_single, encoder_single = Variable(
                            vid_tensor[tcncover - 1]).cuda(), Variable(
                            labels[tcncover - 1]).cuda(), Variable(
                            diffs[tcncover - 1]).cuda(), Variable(exes[tcncover - 1]).cuda(), Variable(
                            encs[tcncover - 1]).cuda()
                        labels_single = labels_single[:, np.newaxis]
                        diffs_single = diffs_single[:, np.newaxis]
                        exe_single = exe_single[:, np.newaxis]
                        encoder_single = encoder_single[:, np.newaxis]
                    else:
                        vid_tensor, labels = Variable(vid_tensor), Variable(labels)
                    model.eval()

                    test_output = model(vid_tensor_single)
                    test_output = torch.cat((test_output, encoder_single), 1)
                    tfc_model1.eval()
                    tfc_model2.eval()
                    tfc_model3.eval()
                    tfc_model4.eval()
                    tfc_model5.eval()
                    tfc_model = eval('tfc_model' + str(tcncover))
                    test_output = tfc_model(test_output)
                    vid_tensor_save.append(test_output)
                    ## [2,5]
                test_output_sum = torch.cat((vid_tensor_save[0], vid_tensor_save[1],
                                             vid_tensor_save[2], vid_tensor_save[3], vid_tensor_save[4]), 1)
                sub_output = vid_tensor_save

                # stage_score1 = sub_output[0]
                # stage_score2 = sub_output[1]
                # stage_score3 = sub_output[2]
                # stage_score4 = sub_output[3]
                # stage_score5 = sub_output[4]
                stage_score1 = torch.sigmoid(sub_output[0])
                stage_score2 = torch.sigmoid(sub_output[1])
                stage_score3 = torch.sigmoid(sub_output[2])
                stage_score4 = torch.sigmoid(sub_output[3])
                stage_score5 = torch.sigmoid(sub_output[4])

                # f = open('./results/sub_Scores_stage.txt',mode='a')


                end = int(start + options.batch_size)
                j_range = range(0,options.batch_size)
                i_range = range(start,end)
                # j = 0
                ## if extract training_sub, uncomment this
                for (i,j) in zip(i_range,j_range):
                        all_training_samples_subscores[i].append(int(video_name[j]))
                        all_training_samples_subscores[i].append(stage_score1[j].data.cpu().numpy())
                        all_training_samples_subscores[i].append(stage_score2[j].data.cpu().numpy())
                        all_training_samples_subscores[i].append(stage_score3[j].data.cpu().numpy())
                        all_training_samples_subscores[i].append(stage_score4[j].data.cpu().numpy())
                        all_training_samples_subscores[i].append(stage_score5[j].data.cpu().numpy())

                # ## if extract testing subs, uncomment this
                # for (i,j) in zip(i_range,j_range):
                #         all_test_samples_subscores[i].append(int(video_name[j]))
                #         all_test_samples_subscores[i].append(stage_score1[j].data.cpu().numpy())
                #         all_test_samples_subscores[i].append(stage_score2[j].data.cpu().numpy())
                #         all_test_samples_subscores[i].append(stage_score3[j].data.cpu().numpy())
                #         all_test_samples_subscores[i].append(stage_score4[j].data.cpu().numpy())
                #         all_test_samples_subscores[i].append(stage_score5[j].data.cpu().numpy())

                ttc_model.eval()

                exe_score_output = ttc_model(test_output_sum) * 30
                test_output_sum = exe_score_output * diffs_single

                all_exes_output = np.append(all_exes_output, exe_score_output.data.cpu().numpy()[:, 0])
                all_exescores = np.append(all_exescores, exe_single.data.cpu().numpy())

                all_test_output = np.append(all_test_output, test_output_sum.data.cpu().numpy()[:, 0])
                all_labels = np.append(all_labels, labels_single.data.cpu().numpy())

                for i in range(len(labels_single.data.cpu().numpy())):
                  f = open('./results/layer_ablation(1FC)/res_for_training_subs.txt', mode='a')
                  # f = open('./results/layer_ablation(1FC)/res_for_testing_subs.txt', mode='a')
                  f.write(
                      '\n' + str(test_output_sum[i].data.cpu().numpy()) + '-' + str(
                          labels_single.data.cpu().numpy()[i]))
                  f.write('\n' + str(exe_score_output.data.cpu().numpy()[i]) + '-' + str(
                      exe_single.data.cpu().numpy()[i]))
                  f.write('\n')

          # np.save('./results/layer_ablation(1FC)/testing_sub_scores.npy',all_test_samples_subscores)
          np.save('./results/layer_ablation(1FC)/training_sub_scores.npy',all_training_samples_subscores)


          rho1, p_val1 = spearmanr(all_test_output, all_labels)
          rho2, p_val2 = spearmanr(all_exes_output, all_exescores)
          n = len(all_labels)
          mse1 = sum(np.square(all_labels - all_test_output)) / n
          mse2 = sum(np.square(all_exescores - all_exes_output)) / n

          logging.info(
              "The overall-score corr is {0} , the execution scores corr is {1},the overall-score mse is {2} , the execution scores mse is {3}.".format(
                  rho1, rho2, mse1, mse2))


if __name__ == "__main__":
    ret = parser.parse_known_args()
    options = ret[0]
    if ret[1]:
        logging.warning("unknown arguments: {0}".format(parser.parse_known_args()[1]))
    main(options)
