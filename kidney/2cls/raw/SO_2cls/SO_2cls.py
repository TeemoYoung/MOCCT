from sklearn.model_selection import StratifiedKFold
import pandas as pd
from os import path
import os
import torch
from torch.utils.data import TensorDataset, DataLoader
from Classes.modelclasses import MOCCT, BP
from torch import nn
from torch.optim import lr_scheduler
from Classes.trainfunction import MOCCT_fit, BP_fit
from sklearn import metrics
import copy
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.svm import SVC
from sklearn.naive_bayes import ComplementNB

# Build a list of omics categories and a list of file paths
omics_list = ["mRNA", "miRNA"]
mRNApath = "./kidney_2cls_mRNA_rf.csv"
miRNApath = "./kidney_2cls_miRNA_rf.csv"
files_list = [mRNApath, miRNApath]

# cross validation
skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
# batchsize
batch_size = 128

# save path
savepath = "./result"
if not path.isdir(savepath):
    os.makedirs(savepath)
    pass

def metric_cv_mean():
    model_name = ['MOCCT', 'BP', 'DT', 'SVM', 'NB']
    acc_list = [(fold_train_acc, fold_test_acc),
                (fold_train_acc_bp, fold_test_acc_bp),
                (fold_train_acc_dt, fold_test_acc_dt),
                (fold_train_acc_svm, fold_test_acc_svm),
                (fold_train_acc_nb, fold_test_acc_nb)]
    recall_list = [(fold_train_recall, fold_test_recall),
                   (fold_train_recall_bp, fold_test_recall_bp),
                   (fold_train_recall_dt, fold_test_recall_dt),
                   (fold_train_recall_svm, fold_test_recall_svm),
                   (fold_train_recall_nb, fold_test_recall_nb)]
    f1_list = [(fold_train_f1, fold_test_f1),
               (fold_train_f1_bp, fold_test_f1_bp),
               (fold_train_f1_dt, fold_test_f1_dt),
               (fold_train_f1_svm, fold_test_f1_svm),
               (fold_train_f1_nb, fold_test_f1_nb)]

    acc_cv_list, recall_cv_list, f1_cv_list = [], [], []
    for name, acc, recall, f1 in zip(model_name, acc_list, recall_list, f1_list):
        acc_train_mean = sum(acc[0]) / len(acc[0])
        acc_test_mean = sum(acc[1]) / len(acc[1])
        print("_" * 50,
              f"\n{name}CV accuracy(train)：", round(acc_train_mean, 3),
              f"\n{name}CV accuracy(test)：", round(acc_test_mean, 3))
        recall_train_mean = sum(recall[0]) / len(recall[0])
        recall_test_mean = sum(recall[1]) / len(recall[1])
        print("_" * 50,
              f"\n{name}CV recall(train)：", round(recall_train_mean, 3),
              f"\n{name}CV recall(test)：", round(recall_test_mean, 3))
        f1_train_mean = sum(f1[0]) / len(f1[0])
        f1_test_mean = sum(f1[1]) / len(f1[1])
        print("_" * 50,
              f"\n{name}CV F1score(train)：", round(f1_train_mean, 3),
              f"\n{name}CV F1score(test)：", round(f1_test_mean, 3))


        acc_cv_list.append(acc_test_mean)
        recall_cv_list.append(recall_test_mean)
        f1_cv_list.append(f1_test_mean)
    return acc_cv_list, recall_cv_list, f1_cv_list


for omics, datafile in zip(omics_list, files_list):
    # create metrics table
    metrics_table = pd.DataFrame(index=['MOCCT', 'BP', 'DT', 'SVM', 'NB'])

    print("*" * 20 + " " + omics + " " + "*" * 20) 
    # read data
    data = pd.read_csv(datafile, index_col=0)
    # check data
    print(f'{omics} sample counts：', data.shape[0])
    print(f'{omics} feature counts：', data.shape[1] - 1)
  
    # train data X
    X = data.iloc[:, :-1].values
    # label Y
    Y = data.label.values.reshape(-1, 1)

    # MOCCT result mean list
    fold_train_acc, fold_test_acc = [], []
    fold_train_recall, fold_test_recall = [], []
    fold_train_f1, fold_test_f1 = [], []

    # BP result mean list
    fold_train_acc_bp, fold_test_acc_bp = [], []
    fold_train_recall_bp, fold_test_recall_bp = [], []
    fold_train_f1_bp, fold_test_f1_bp = [], []

    # DT result mean list
    fold_train_acc_dt, fold_test_acc_dt = [], []
    fold_train_recall_dt, fold_test_recall_dt = [], []
    fold_train_f1_dt, fold_test_f1_dt = [], []

    # SVM result mean list
    fold_train_acc_svm, fold_test_acc_svm = [], []
    fold_train_recall_svm, fold_test_recall_svm = [], []
    fold_train_f1_svm, fold_test_f1_svm = [], []

    # NB result mean list
    fold_train_acc_nb, fold_test_acc_nb = [], []
    fold_train_recall_nb, fold_test_recall_nb = [], []
    fold_train_f1_nb, fold_test_f1_nb = [], []

    for i in ["MOCCT", "BP", "DT", "SVM", "NB"]:
        omicsfolder = os.path.join(savepath, omics, i)
        if not path.isdir(omicsfolder):
            os.makedirs(omicsfolder)
            pass
        pass

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, Y)):
        x_train, x_test = X[train_idx], X[test_idx]
        y_train, y_test = Y[train_idx], Y[test_idx]

        # check
        print("CV {} ".format(fold + 1))
        print("train X：", x_train.shape, "test X：", x_test.shape)
        print("train Y：", y_train.shape, "test Y：", y_test.shape)

        # tensor
        x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
        x_test = torch.from_numpy(x_test).type(torch.FloatTensor)
        y_train = torch.from_numpy(y_train).type(torch.FloatTensor)
        y_test = torch.from_numpy(y_test).type(torch.FloatTensor)

        train_ds = TensorDataset(x_train, y_train)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        test_ds = TensorDataset(x_test, y_test)
        test_dl = DataLoader(test_ds, batch_size=batch_size)

        """
        MOCCT
        """
        print("-" * 20 + f"MOCCT-CV {fold + 1} START" + "-" * 20)
        mocct = MOCCT()
        lr = 0.00001
        optim = torch.optim.Adam(mocct.parameters(), lr=lr)
        loss_fn = nn.BCELoss()
        epochs = 2000
        scheduler = lr_scheduler.StepLR(optim, step_size=2000, gamma=0.1)

        # loss and acc list
        train_loss, train_acc = [], []
        test_loss, test_acc = [], []

        for epoch in range(epochs):
            
            best_mocct_wts = copy.deepcopy(mocct.state_dict())
            best_acc = 0.0

            epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc = MOCCT_fit(epoch,
                                                                               mocct,
                                                                               train_dl,
                                                                               test_dl,
                                                                               optim,
                                                                               loss_fn,
                                                                               scheduler)
            if epoch_test_acc > best_acc:
                best_acc = epoch_test_acc
                best_mocct_wts = copy.deepcopy(mocct.state_dict())

                pass
            
            train_loss.append(epoch_loss)
            train_acc.append(epoch_acc)

            test_loss.append(epoch_test_loss)
            test_acc.append(epoch_test_acc)

            pass

        # best model
        mocct.load_state_dict(best_mocct_wts)

        y_pred_train = (mocct(x_train, x_train, x_train) > 0.5).type(torch.int32)
        y_pred_test = (mocct(x_test, x_test, x_test) > 0.5).type(torch.int32)

        # mocct accuracy
        acc_train = metrics.accuracy_score(y_train, y_pred_train)
        acc_test = metrics.accuracy_score(y_test, y_pred_test)
        print("MOCCT-train accuracy：", round(acc_train, 3), "\nMOCCT-test accuracy：", round(acc_test, 3))
        fold_train_acc.append(acc_train)
        fold_test_acc.append(acc_test)

        # mocct recall
        recall_train = metrics.recall_score(y_train, y_pred_train, average='macro')
        recall_test = metrics.recall_score(y_test, y_pred_test, average='macro')
        print("MOCCT-train recall：", round(recall_train, 3), "\nMOCCT-test recall：", round(recall_test, 3))
        fold_train_recall.append(recall_train)
        fold_test_recall.append(recall_test)

        # mocct F1score
        f1_train = metrics.f1_score(y_train, y_pred_train, average='macro')
        f1_test = metrics.f1_score(y_test, y_pred_test, average='macro')
        print("MOCCT-train F1score：", round(f1_train, 3), "\nMOCCT-test F1score：", round(f1_test, 3))
        fold_train_f1.append(f1_train)
        fold_test_f1.append(f1_test)

        # loss plot
        plt.plot(range(1, epochs + 1), train_loss, label='train_loss')
        plt.plot(range(1, epochs + 1), test_loss, label='test_loss')
        plt.title(f"MOCCT fold {fold + 1} Loss feature-num = {30}")
        plt.legend()
        plt.savefig(os.path.join(savepath, "MOCCT", f"MOCCT-fold{fold + 1}-loss-{30}.jpg"))
        plt.clf()

        # acc plot
        plt.plot(range(1, epochs + 1), train_acc, label='train_acc')
        plt.plot(range(1, epochs + 1), test_acc, label='test_acc')
        plt.title(f"MOCCT fold {fold + 1} Acc feature-num = {30}")
        plt.legend()
        plt.savefig(os.path.join(savepath, "MOCCT", f"MOCCT-fold{fold + 1}-acc-{30}.jpg"))
        plt.clf()
        
        print("-" * 20 + f"MOCCT-CV {fold + 1} END" + "-" * 20)


        """
        BP
        """
        print("-" * 20 + f"BP-CV {fold + 1} START" + "-" * 20)
        bp = BP()
        lr = 0.00001
        optim_bp = torch.optim.Adam(bp.parameters(), lr=lr)
        loss_fn_bp = nn.BCELoss()
        epochs = 2000
        scheduler_bp = lr_scheduler.StepLR(optim_bp, step_size=2000, gamma=0.1)

        # loss and acc list
        train_loss_bp, train_acc_bp = [], []
        test_loss_bp, test_acc_bp = [], []

        for epoch in range(epochs):
           
            best_bp_wts = copy.deepcopy(bp.state_dict())
            best_acc = 0.0

            epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc = BP_fit(epoch, 
                                                                            bp, 
                                                                            train_dl, 
                                                                            test_dl,
                                                                            optim_bp,
                                                                            loss_fn_bp,
                                                                            scheduler_bp)
            if epoch_test_acc > best_acc:
                best_acc = epoch_test_acc
                best_bp_wts = copy.deepcopy(bp.state_dict())
                
                pass
            
            train_loss_bp.append(epoch_loss)
            train_acc_bp.append(epoch_acc)

            test_loss_bp.append(epoch_test_loss)
            test_acc_bp.append(epoch_test_acc)

            pass

        # best model
        bp.load_state_dict(best_bp_wts)

        y_pred_train_bp = (bp(x_train) > 0.5).type(torch.int32)
        y_pred_test_bp = (bp(x_test) > 0.5).type(torch.int32)

        # bp accuracy
        acc_train_bp = metrics.accuracy_score(y_train, y_pred_train_bp)
        acc_test_bp = metrics.accuracy_score(y_test, y_pred_test_bp)
        print("BP-train accuracy：", round(acc_train_bp, 3), "BP-test accuracy：", round(acc_test_bp, 3))
        fold_train_acc_bp.append(acc_train_bp)
        fold_test_acc_bp.append(acc_test_bp)

        # bp recall
        recall_train_bp = metrics.recall_score(y_train, y_pred_train_bp, average="macro")
        recall_test_bp = metrics.recall_score(y_test, y_pred_test_bp, average="macro")
        print("BP-train recall：", round(recall_train_bp, 3), "BP-test recall：", round(recall_test_bp, 3))
        fold_train_recall_bp.append(recall_train_bp)
        fold_test_recall_bp.append(recall_test_bp)

        # bp F1score
        f1_train_bp = metrics.f1_score(y_train, y_pred_train_bp, average='macro')
        f1_test_bp = metrics.f1_score(y_test, y_pred_test_bp, average='macro')
        print("BP-train F1score：", round(f1_train_bp, 3), "BP-test F1score：", round(f1_test_bp, 3))
        fold_train_f1_bp.append(f1_train_bp)
        fold_test_f1_bp.append(f1_test_bp)

        # loss plot
        plt.plot(range(1, epochs + 1), train_loss_bp, label='train_loss')
        plt.plot(range(1, epochs + 1), test_loss_bp, label='test_loss')
        plt.title("BP fold {} Loss feature-num = {}".format(fold + 1, 50))
        plt.legend()
        plt.savefig(os.path.join(savepath, "BP", "bp-fold{}-loss-i-{}.jpg".format(fold + 1, 50)))
        plt.clf()

        # acc plot
        plt.plot(range(1, epochs + 1), train_acc_bp, label='train_acc')
        plt.plot(range(1, epochs + 1), test_acc_bp, label='test_acc')
        plt.title("Net fold {} Accuracy feature-num = {}".format(fold + 1, 50))
        plt.legend()
        plt.savefig(os.path.join(savepath, "BP", "bp-fold{}-accuracy-i-{}.jpg".format(fold + 1, 50)))
        plt.clf()
        print("-" * 20 + f"BP-CV {fold + 1} END" + "-" * 20)

        """
        DT
        """
        print("-" * 20 + f"DT-CV {fold + 1} START" + "-" * 20)
        
        dt = tree.DecisionTreeClassifier()
        dt = dt.fit(x_train.numpy(), y_train.numpy().reshape(-1))

        y_pred_train_dt = dt.predict(x_train.numpy())
        y_pred_test_dt = dt.predict(x_test.numpy())

        # dt accuracy
        acc_train_dt = metrics.accuracy_score(y_train, y_pred_train_dt)
        acc_test_dt = metrics.accuracy_score(y_test, y_pred_test_dt)
        print("DT-train accuracy：", round(acc_train_dt, 3), "DT-test accuracy：", round(acc_test_dt, 3))
        fold_train_acc_dt.append(acc_train_dt)
        fold_test_acc_dt.append(acc_test_dt)

        # dt recall
        recall_train_dt = metrics.recall_score(y_train, y_pred_train_dt, average='macro')
        recall_test_dt = metrics.recall_score(y_test, y_pred_test_dt, average='macro')
        print("DT-train recall：", round(recall_train_dt, 3), "DT-test recall：", round(recall_test_dt, 3))
        fold_train_recall_dt.append(recall_train_dt)
        fold_test_recall_dt.append(recall_test_dt)

        # dt F1score
        f1_train_dt = metrics.f1_score(y_train, y_pred_train_dt, average='macro')
        f1_test_dt = metrics.f1_score(y_test, y_pred_test_dt, average='macro')
        print("DT-train F1score：", round(f1_train_dt, 3), "DT-test F1score：", round(f1_test_dt, 3))
        fold_train_f1_dt.append(f1_train_dt)
        fold_test_f1_dt.append(f1_test_dt)
        print("-" * 20 + f"DT-CV {fold + 1} END" + "-" * 20)

        """
        SVM
        """
        print("-" * 20 + f"SVM-CV {fold + 1} START" + "-" * 20)
        svm = SVC(kernel="sigmoid")
        svm = svm.fit(x_train.numpy(), y_train.numpy().reshape(-1))

        y_pred_train_svm = svm.predict(x_train.numpy())
        y_pred_test_svm = svm.predict(x_test.numpy())

        # svm accuracy
        acc_train_svm = metrics.accuracy_score(y_train, y_pred_train_svm)
        acc_test_svm = metrics.accuracy_score(y_test, y_pred_test_svm)
        print("SVM-train accuracy：", round(acc_train_svm, 3), "SVM-test accuracy：", round(acc_test_svm, 3))
        fold_train_acc_svm.append(acc_train_svm)
        fold_test_acc_svm.append(acc_test_svm)

        # svm recall
        recall_train_svm = metrics.recall_score(y_train, y_pred_train_svm, average='macro')
        recall_test_svm = metrics.recall_score(y_test, y_pred_test_svm, average='macro')
        print("SVM-train recall：", round(recall_train_svm, 3), "SVM-test recall：", round(recall_test_svm, 3))
        fold_train_recall_svm.append(recall_train_svm)
        fold_test_recall_svm.append(recall_test_svm)

        # svm F1score
        f1_train_svm = metrics.f1_score(y_train, y_pred_train_svm, average='macro')
        f1_test_svm = metrics.f1_score(y_test, y_pred_test_svm, average='macro')
        print("SVM-train F1score：", round(f1_train_svm, 3), "SVM-test F1score：", round(f1_test_svm, 3))
        fold_train_f1_svm.append(f1_train_svm)
        fold_test_f1_svm.append(f1_test_svm)
        print("-" * 20 + f"SVM-CV {fold + 1} END" + "-" * 20)

        """
        NB
        """
        print("-" * 20 + f"NB-CV {fold + 1} START" + "-" * 20)
        
        nb = ComplementNB()
        nb = nb.fit(x_train.numpy(), y_train.numpy().reshape(-1))

        y_pred_train_nb = nb.predict(x_train.numpy())
        y_pred_test_nb = nb.predict(x_test.numpy())

        # nb accuracy
        acc_train_nb = metrics.accuracy_score(y_train, y_pred_train_nb)
        acc_test_nb = metrics.accuracy_score(y_test, y_pred_test_nb)
        print("NB-train accuracy：", round(acc_train_nb, 3), "NB-test accuracy：", round(acc_test_nb, 3))
        fold_train_acc_nb.append(acc_train_nb)
        fold_test_acc_nb.append(acc_test_nb)

        # nb recall
        recall_train_nb = metrics.recall_score(y_train, y_pred_train_nb, average='macro')
        recall_test_nb = metrics.recall_score(y_test, y_pred_test_nb, average='macro')
        print("NB-train recall：", round(recall_train_nb, 3), "NB-test recall：", round(recall_test_nb, 3))
        fold_train_recall_nb.append(recall_train_nb)
        fold_test_recall_nb.append(recall_test_nb)

        # nb F1score
        f1_train_nb = metrics.f1_score(y_train, y_pred_train_nb, average='macro')
        f1_test_nb = metrics.f1_score(y_test, y_pred_test_nb, average='macro')
        print("NB-train F1score：", round(f1_train_nb, 3), "NB-test F1score：", round(f1_test_nb, 3))
        fold_train_f1_nb.append(f1_train_nb)
        fold_test_f1_nb.append(f1_test_nb)
        print("-" * 20 + f"NB-CV {fold + 1} END" + "-" * 20)

        pass
    
    # CV result
    acc_cv_list, recal_cv_list, f1_cv_list = metric_cv_mean()

    metrics_table["accuracy"] = acc_cv_list
    metrics_table["recall"] = recal_cv_list
    metrics_table["F1score"] = f1_cv_list

    metrics_table.to_csv(f"{omics}_SO_2cls_metrics_result.csv")

    print("!!! END !!!")