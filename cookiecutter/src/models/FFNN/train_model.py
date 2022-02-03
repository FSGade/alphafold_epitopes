#!/usr/bin/env python3

print("Importing common model setup...")
import sys
sys.path.append("..")
from common_model_setup import *

SOLVED = False
print("Using {} structures for FFNN".format("solved" if SOLVED else "AF2"))

device = "cuda" if torch.cuda.is_available() else "cpu"
print("The device in use is", device)

BATCH_SIZE = 64
MAX_EPOCHS = 200
LR = 1e-3
LR_DECAY = 1e-5

EARLY_STOPPING = True

training_set, val_set, test_set = create_positional_datasets(
        *create_embedding_dicts(SOLVED), device
        )

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.layers = nn.Sequential(
                    nn.BatchNorm1d(128),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(p=0.1),
                    nn.Linear(64, 16),
                    nn.ReLU(),
                    nn.Dropout(p=0.1),
                    nn.Linear(16, 1, bias=False),
                    nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layers(x)
    
net = Net()
net.to(device)

trainloader = torch.utils.data.DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=False)

train_X, train_y = training_set.tensors
test_X, test_y = test_set.tensors
val_X, val_y = val_set.tensors


################
### TRAINING ###
################

criterion = nn.BCELoss(reduction="none")
optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=LR_DECAY)

def weight_reset(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        m.reset_parameters()

net.apply(weight_reset)
losses=[]
training_plot=[]
test_plot=[]
val_plot=[]
auc_train_plot=[]
auc_test_plot=[]
mcc_train_plot=[]
mcc_test_plot=[]

last_score=np.inf
max_es_rounds = 5
es_rounds = max_es_rounds
best_epoch = 0

for epoch in range(MAX_EPOCHS):  # loop over the dataset multiple times
    net.train()

    for i, data in enumerate(trainloader,0):

        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()
        #scheduler.step()
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(torch.flatten(outputs), labels.to(device))
        loss = torch.sum(loss)
        loss.backward()
        optimizer.step()

# print statistics
    with torch.no_grad():
        test_loss=0
        train_loss=0
        net.eval()
        inputs, labels = train_X, train_y

        outputs = net(inputs)

        loss = criterion(torch.flatten(outputs), labels)
        loss = torch.mean(loss)
        training_plot.append(loss.cpu().numpy())
        auc_train_plot.append(roc_auc_score(labels.cpu(), outputs.cpu().flatten()))
        mcc_train_plot.append(mcc(labels.cpu(), outputs.cpu().flatten()>.1))


        inputs, labels = test_X, test_y
        outputs = net(inputs)
        loss = criterion(torch.flatten(outputs), labels)
        loss = torch.mean(loss)
        test_plot.append(loss.cpu().numpy())
        auc_test_plot.append(roc_auc_score(labels.cpu(), outputs.cpu().flatten()))
        fpr, tpr, _ = roc_curve(labels.cpu(), outputs.cpu().flatten())
        mcc_test_plot.append(mcc(labels.cpu(), outputs.cpu().flatten()>.1))

        inputs, labels = val_X, val_y
        outputs = net(inputs)
        valloss = criterion(torch.flatten(outputs), labels)
        valloss = torch.mean(valloss)
        val_plot.append(valloss.cpu().numpy())
        print("Epoch {}, training loss {}, test loss {}, validation loss {}".format(epoch, training_plot[-1], test_plot[-1], valloss))
        print("Epoch {}, training AUC {}, test AUC {}".format(epoch, auc_train_plot[-1], auc_test_plot[-1]))
        print("Epoch {}, training MCC {}, test MCC {}".format(epoch, mcc_train_plot[-1], mcc_test_plot[-1]))

    if EARLY_STOPPING:
        if last_score > valloss:
            last_score = valloss
            best_epoch = epoch
            es_rounds = max_es_rounds
            best_model = copy.deepcopy(net)
            ffnn_fpr, ffnn_tpr = fpr, tpr
        else:
            if es_rounds > 0:
                es_rounds -=1
            else:
                print('EARLY-STOPPING !')
                print('Best epoch found: nº {}'.format(best_epoch))
                print('Exiting. . .')
                break

print("Finished training.")


