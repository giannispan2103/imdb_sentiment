from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import numpy as np
import torch
from torch.autograd import Variable
from time import time
from paths import MODELS_PATH

def train(model, train_batches, test_batches, optimizer, criterion, epochs, init_patience, evaluator='f1', cuda=True):
    """
    :param model: a deep model
    :param train_batches: the batches that will be used for training
    :param test_batches:the batches that will be used for testing
    :param optimizer: the optimization algorithm that used for training
    :param criterion: the loss function (almost always Binary Cross Entropy)
    :param epochs: the max number of epochs
    :param init_patience: the number of epochs that the training will last after
    its best performance in validation data
    """
    best_eval = 0
    patience = init_patience
    for i in range(1, epochs+1):
        start = time()
        val_dict = run_epoch(model, train_batches, test_batches, optimizer, criterion,  cuda)
        end = time()
        print('epoch %d, f1: %2.3f accuracy: %2.3f auc: %2.3f. Time: %d minutes, %d seconds'
              % (i,  100 * val_dict['f1'], 100 * val_dict['acc'], 100 * val_dict['auc'],
                 (end - start) / 60, (end - start) % 60))
        if best_eval < val_dict[evaluator]:
            best_eval = val_dict[evaluator]
            patience = init_patience
            save_model(model)
            if i > 1:
                print('best epoch so far')
        else:
            patience -= 1
        if patience == 0:
            break


def run_epoch(model, train_batches, test_batches, optimizer, criterion, cuda):
    model.train(True)
    perm = np.random.permutation(len(train_batches))
    for i in perm:
        batch = train_batches[i]
        inner_perm = np.random.permutation(len(batch['text']))
        data = []
        for inp in model.input_list:
            if cuda:
                data.append(Variable(torch.from_numpy(batch[inp][inner_perm]).long().cuda()))
            else:
                data.append(Variable(torch.from_numpy(batch[inp][inner_perm]).long()))
        if cuda:
            labels = Variable(torch.from_numpy(batch['label'][inner_perm]).cuda())
        else:
            labels = Variable(torch.from_numpy(batch['label'][inner_perm]))
        outputs = model(*data)
        loss = criterion(outputs.view(-1), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return evaluate(model, test_batches, cuda)


def get_scores_and_labels(model, test_batches, cuda):
    scores_list = []
    labels_list = []
    for batch in test_batches:
        data = []
        for inp in model.input_list:
            if cuda:
                data.append(Variable(torch.from_numpy(batch[inp]).long().cuda()))
            else:
                data.append(Variable(torch.from_numpy(batch[inp]).long()))
        outputs = model(*data)
        outputs = torch.sigmoid(outputs)
        labels_list.extend(batch['sentiment'].tolist())
        scores_list.extend(outputs.data.view(-1).tolist())
    return labels_list, scores_list


def evaluate(model, test_batches, cuda):
    model.train(False)
    labels_list, scores_list = get_scores_and_labels(model, test_batches, cuda)

    return {'auc': roc_auc_score(np.asarray(labels_list, dtype='float32'), np.asarray(scores_list, dtype='float32')),
            'acc': tune_threshold(np.asarray(labels_list, dtype='float32'), np.asarray(scores_list, dtype='float32')),
            'f1': tune_threshold(np.asarray(labels_list, dtype='float32'), np.asarray(scores_list, dtype='float32'), False)}


def tune_threshold(labels, scores, is_acc=True):
    init_thr = 0.25
    final_thr = 0.75
    pace = 0.01
    best = 0.0
    if is_acc:
        evaluator = accuracy_score
    else:
        evaluator = f1_score
    for thr in np.arange(init_thr, final_thr, pace):
        acc = evaluator(np.asarray(labels, dtype='float32'), np.asarray([x > thr for x in scores], dtype='float32'))
        if acc > best:
            best = acc
    return best


def save_model(model):
    torch.save(model.state_dict(), MODELS_PATH + model.name + '.pkl')


def load_model(model):
    model.load_state_dict(torch.load(MODELS_PATH + model.name + '.pkl'))