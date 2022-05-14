import torch.optim as optim
import torch
import torch.nn as nn
from sklearn.model_selection import KFold 
from sklearn.metrics import roc_curve, auc
import torch.optim as optim
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix, precision_recall_curve, f1_score
import numpy as np

def train_model(model, train_x, train_y, train_demo = None, max_epochs = 20, batch_size = 64, lr = 0.00025):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    opt = optim.Adam(model.parameters(), lr=lr)
    weights = torch.FloatTensor([3.125])
    bce = nn.BCEWithLogitsLoss(pos_weight = weights).to(device)
    for epoch in range(max_epochs):
        i = 0
        while i < len(train_x):
            if i + batch_size < len(train_x):
                end = i + batch_size
            else:
                end = len(train_x)-1
            x = train_x[i:end]
            y = train_y[i:end]
            if train_demo is not None:
                demo = train_demo[i:end]
                out = model(x, demo).squeeze()
            else:
                out = model(x).squeeze()
            y = y.squeeze()
            loss = bce(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            i += batch_size
    return model

def auc_(model, x, y):
    sigmoid = nn.Sigmoid()
    outputs = sigmoid(model(x))
    fpr, tpr, _ = roc_curve(y.cpu().detach().numpy(), outputs.cpu().detach().numpy())
    roc_auc = auc(fpr, tpr)
    return roc_auc, fpr, tpr

def convertTuple(tup):
    string = ''
    for item in tup:
        string = string + str(item) + ", "
    return string

def gridSearchCV(train_x, train_y, untrained_model, lrs = [0.0001, 0.00025, 0.0005], epochs = [30, 20, 50], batch_sizes = [64]):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Doing", len(lrs)*len(epochs)*len(batch_sizes)*5, "trainings...")
    print()
    results = {}
    i = 0
    for lr in lrs:
        for epoch in epochs:
            for batch_size in batch_sizes:
                tag = "lr=" + str(lr) + ", epochs=" + str(epoch) + ", batch_size=" + str(batch_size)
                # do five fold cross validation
                kf = KFold(n_splits=5)
                auc_list = []
                for train_index, test_index in kf.split(train_x):
                    X_train, X_test = train_x[train_index].to(device), train_x[test_index].to(device)
                    y_train, y_test = train_y[train_index].to(device), train_y[test_index].to(device)
                    trained_model = train_model(untrained_model, X_train, y_train, lr = lr, max_epochs = epoch,\
                                                     batch_size = batch_size)
                    roc_auc, fpr, tpr = auc_(trained_model, X_test, y_test)
                    auc_list.append(roc_auc)
                    print("Training", i, "complete:", round(roc_auc, 2))
                    i +=1
                avg_auc = round(sum(auc_list)/len(auc_list), 2)
                results[tag] = avg_auc
            
    return results

def full_evaluation(untrained_model, train_x, train_y, test_x, test_y, train_demo = None, test_demo = None):
    print("Evaluating model...")
    trained_model = train_model(untrained_model, train_x, train_y, train_demo=train_demo, lr = 0.0075, max_epochs=20, \
                                batch_size=32)
    trained_model.eval()
    aucs = []
    accs = []
    f1_scores = []
    precisions = []
    sensitivities = []
    specificitys = []
    npvs = []

    with torch.no_grad():
        for i in range(100):
            sample_x = resample(test_x, replace=True, random_state = i) # random_state = i to ensure sample_x and sample_y match
            sample_y = resample(test_y, replace=True, random_state = i).cpu().detach().numpy()
            sigmoid = nn.Sigmoid()
            if train_demo is not None:
                sample_demo = resample(test_demo, replace=True, random_state = i)
                outputs = sigmoid(trained_model(sample_x, sample_demo)).cpu().detach().numpy()
            else:
                outputs = sigmoid(trained_model(sample_x)).cpu().detach().numpy()
            rounded_outputs = np.array([np.round(output) for output in outputs]).squeeze()
            fpr, tpr, _ = roc_curve(sample_y, outputs)
            roc_auc = auc(fpr, tpr)
            precision, recall, thresholds = precision_recall_curve(sample_y, outputs)
            acc = torch.sum(torch.from_numpy(rounded_outputs) == torch.from_numpy(sample_y))
            acc = acc.type(torch.DoubleTensor)/len(sample_y)
            f1 = f1_score(sample_y, rounded_outputs)
            tn, fp, fn, tp = confusion_matrix(sample_y, rounded_outputs).ravel()
            specificity = tn / (tn+fp)
            sensitivity = tp / (tp + fn)
            precision = tp / (tp + fp)

            # skip calculating npv for samples without any tn and tp to avoid 0/0:
            if tn != 0 and tp != 0: 
                npv = tn / (tn + fn)
                npvs.append(npv)

            aucs.append(roc_auc)
            precisions.append(precision)
            accs.append(acc)
            f1_scores.append(f1)
            specificitys.append(specificity)
            sensitivities.append(sensitivity)
            
        metrics = [aucs, accs, f1_scores, precisions, sensitivities, specificitys, npvs]
        medians = []
        percent_2_5 = []
        percent_97_5 = []
        for metric in metrics:
            medians.append(np.percentile(metric, 50))
            percent_2_5.append(np.percentile(metric, 2.5))
            percent_97_5.append(np.percentile(metric, 97.5))
        medians = [round(x, 2) for x in medians]
        percent_2_5 = [round(x, 2) for x in percent_2_5]
        percent_97_5 = [round(x, 2) for x in percent_97_5]

        format_row = "{:<10}" * (len(medians) + 1)
        print(format_row.format("PCTL", 'AUC','Acc.','F1','Prec.', "Sens.", "Spec.", "NPV"))
        print(format_row.format("50%", *medians))
        print(format_row.format("2.5%", *percent_2_5))
        print(format_row.format("97.5%", *percent_97_5))