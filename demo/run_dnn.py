from torch.utils.data import DataLoader
from dnn_dataset import dnn_dataset
import torch
from dnn import dnn
from torch import optim
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from scipy.stats import bootstrap
import numpy as np

train_data = dnn_dataset('train', '../Stanford_Extract/checkpoints/inputs_xgboost.csv', '../Stanford_Extract/checkpoints/labels.csv')
train_dataloader = DataLoader(train_data, batch_size = 8, shuffle = True)

eval_data = dnn_dataset('eval', '../Stanford_Extract/checkpoints/inputs_xgboost.csv', '../Stanford_Extract/checkpoints/labels.csv')
eval_dataloader = DataLoader(eval_data, batch_size = 8, shuffle = True)

test_data = dnn_dataset('test', '../Stanford_Extract/checkpoints/inputs_xgboost.csv', '../Stanford_Extract/checkpoints/labels.csv')
test_dataloader = DataLoader(test_data, batch_size = 8, shuffle = True)

model = dnn()
optimizer = optim.Adam(params = model.parameters(), lr = 1e-5)
criterion = torch.nn.BCELoss()

epochs = 6

max_auroc = 0

for epoch in range(epochs):
    train_outputs = []
    train_labels = []
    for data in train_dataloader:
        optimizer.zero_grad()
        model.train()
        inputs, labels = data
        outputs = model(inputs.float())
        train_outputs.extend(list(outputs.detach().numpy()[:, 0]))
        train_labels.extend(list(labels.numpy()))
        loss = criterion(outputs, labels.reshape((-1, 1)).float())
        loss.backward()
        optimizer.step()

    model.eval()
    eval_outputs = []
    eval_labels = []
    for data in eval_dataloader:
        inputs, labels = data
        outputs = model(inputs.float())
        eval_outputs.extend(list(outputs.detach().numpy()[:, 0]))
        eval_labels.extend(list(labels.numpy()))

    val_auroc = roc_auc_score(eval_labels, eval_outputs)
    val_auprc = average_precision_score(eval_labels, eval_outputs)

    print(f"Epoch: {epoch + 1}")
    print(f"Eval AUROC: {val_auroc}")
    print(f"Eval AUPRC: {val_auprc}")

    if val_auroc > max_auroc:
        max_auroc = val_auroc
        torch.save(model.state_dict(), './best_dnn.pt')


#print(f"Mean auroc: {np.mean(max_aurocs)}. Std auroc: {np.std(max_aurocs)}")
#print(f"Mean auprc: {np.mean(max_auprcs)}. Std auprc: {np.std(max_auprcs)}")

model = dnn()
model.load_state_dict(torch.load('./best_dnn.pt'))
model.eval()
test_outputs = []
test_labels = []

for data in test_dataloader:
    inputs, labels = data
    outputs = model(inputs.float())
    test_outputs.extend(list(outputs.detach().numpy()[:, 0]))
    test_labels.extend(list(labels.numpy()))

auroc = roc_auc_score(test_labels, test_outputs)
auprc = average_precision_score(test_labels, test_outputs)

data = (test_labels, test_outputs)

rng = np.random.default_rng()
auroc_ci = bootstrap(data, roc_auc_score, vectorized = False, paired = True, random_state = rng)
auprc_ci = bootstrap(data, average_precision_score, vectorized = False, paired = True, random_state = rng)

print ("AUROC on the test set is {:.3f}".format(auroc))
print(auroc_ci.confidence_interval)
print ("AUPRC on the test set is {:.3f}".format(auprc))
print(auprc_ci.confidence_interval)




