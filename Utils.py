import os
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve
from scipy.special import softmax


class Metrics(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def comp_acc(output, target):
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = output.max(1)
        correct = pred.eq(target).sum()

        return correct * 100.0 / batch_size

def comp_roc(output,target):
    with torch.no_grad():
        prob = output[:,1]
        auc = roc_auc_score(target, prob)
        return auc

def comp_pr(output,target):
    with torch.no_grad():
        prob = output[:,1]
        precision, recall, thresholds = precision_recall_curve(target, prob)
        return auc(recall,precision)

def train(model, device, data_loader, criterion, optimizer, epoch, print_freq=10):
    batch_time = Metrics()
    data_time = Metrics()
    losses = Metrics()
    accuracy = Metrics()
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if isinstance(input, tuple):
            input = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input])
        else:
            input = input.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        assert not np.isnan(loss.item()), 'Model diverged with loss = NaN'

        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        losses.update(loss.item(), target.size(0))
        accuracy.update(comp_acc(output, target).item(), target.size(0))

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})\t'.format(
                epoch, i, len(data_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, acc=accuracy))

    return losses.avg, accuracy.avg


def evaluate(model, device, data_loader, criterion, print_freq=10):
    batch_time = Metrics()
    losses = Metrics()
    accuracy = Metrics()
    results = []

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(data_loader):

            if isinstance(input, tuple):
                input = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input])
            else:
                input = input.to(device)
            target = target.to(device)

            output = model(input)
            loss = criterion(output, target)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            losses.update(loss.item(), target.size(0))
            accuracy.update(comp_acc(output, target).item(), target.size(0))

            y_true = target.detach().to('cpu').numpy().tolist()
            y_pred = output.detach().to('cpu').max(1)[1].numpy().tolist()
            results.extend(list(zip(y_true, y_pred)))

            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})\t'.format(
                    i, len(data_loader), batch_time=batch_time, loss=losses, acc=accuracy))

    return losses.avg, accuracy.avg, results


def predict_task(model, device, data_loader):
    model.eval()
    probas = []
    targets = []
    with torch.no_grad():
        for data in data_loader:
            input, target = data
            if isinstance(input, tuple):
                input = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input])
            else:
                input = input.to(device)
            # get the inputs
            outputs = model(input)
            soft = nn.Softmax(dim=0)
            prob = soft(outputs[0])
            probas.append(prob[1].item())
            targets.append(target)
    # probas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    return probas, targets


