import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from models import TCoN
from plots import plot_curves
from etl import load_data, calculate_num_features, VisitSeqLabelDataset, dl_collate_fn
from Utils import train, evaluate, predict_task, comp_roc, comp_pr


def main():
    # Path for each file
    PATH_TRAIN_FILE = "./data/train_data/"
    PATH_VALID_FILE = "./data/validation_data/"
    PATH_TEST_FILE = "./data/test_data/"

    # Model Variables
    EPOCHS = 15
    BATCH_SIZE = 32
    CUDA = False  # Set 'True' if you want to use GPU
    WORKERS = 0

    device = torch.device("cuda" if torch.cuda.is_available() and CUDA else "cpu")

    # Specify task 1.Mortality 2.Readmission 3.Heart Failure 4.Sepsis
    TASK = 1

    # Path for saving model
    PATH_OUTPUT = "./output/" + str(TASK) + '/'
    os.makedirs(PATH_OUTPUT, exist_ok=True)

    # Load Training data
    train_ids, train_labels, train_seqs = load_data(path=PATH_TRAIN_FILE, task=TASK, mode='train')
    valid_ids, valid_labels, valid_seqs = load_data(path=PATH_VALID_FILE, task=TASK, mode='validation')
    test_ids, test_labels, test_seqs = load_data(path=PATH_TEST_FILE, task=TASK, mode='test')

    num_features = calculate_num_features(train_seqs)

    train_dataset = VisitSeqLabelDataset(train_seqs, train_labels, num_features)
    valid_dataset = VisitSeqLabelDataset(valid_seqs, valid_labels, num_features)
    test_dataset = VisitSeqLabelDataset(test_seqs, test_labels, num_features)

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=dl_collate_fn,
                              num_workers=WORKERS)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=dl_collate_fn,
                              num_workers=WORKERS)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, collate_fn=dl_collate_fn,
                             num_workers=WORKERS)

    model = TCoN(num_features)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=0.001,betas=(0.9,0.999))

    best_val_acc = 0.0
    train_losses, train_accuracies,  = [], [],
    valid_losses, valid_accuracies, = [], [],
    for epoch in range(EPOCHS):
        train_loss, train_accuracy,  = train(model, device, train_loader, criterion, optimizer, epoch)
        valid_loss, valid_accuracy, valid_results = evaluate(model, device, valid_loader, criterion)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        train_accuracies.append(train_accuracy)
        valid_accuracies.append(valid_accuracy)

        is_best = valid_accuracy > best_val_acc  # let's keep the model that has the best accuracy, but you can also use another metric.
        if is_best:
            best_val_acc = valid_accuracy
            torch.save(model, os.path.join(PATH_OUTPUT, "TCoN.pth"), _use_new_zipfile_serialization=False)

    best_model = torch.load(os.path.join(PATH_OUTPUT, "TCoN.pth"))
    plot_curves(train_losses, valid_losses, train_accuracies, valid_accuracies)

    test_prob, test_labels = predict_task(best_model, device, test_loader)

    roc = comp_roc(test_prob,test_labels)
    pr = comp_pr(test_prob,test_labels)

    print(roc, pr)




if __name__ == '__main__':
    main()
