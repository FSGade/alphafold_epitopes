#!/usr/bin/env python3

print("Importing common model setup...")
import sys
sys.path.append("..")
from common_model_setup import *

SOLVED = False
print("Using {} structures for CNN".format("solved" if SOLVED else "AF2"))

device = "cuda" if torch.cuda.is_available() else "cpu"
print("The device in use is", device)

BATCH_SIZE = 8
LR = 5e-3
LR_DECAY = 1e-5
DROPOUT = 0.5
HID_CHANNELS = 75
MAX_EPOCHS = 200
EARLY_STOPPING = True

training_set, val_set, test_set = create_sequential_datasets(
        *create_embedding_dicts(SOLVED), device
        )

in_channels = 128

class Simple1DCNN(torch.nn.Module):
    def __init__(self):
        super(Simple1DCNN, self).__init__()
        self.bn_input = nn.BatchNorm1d(in_channels)
        
        # Conv1
        self.conv1 = nn.Sequential(
                                    nn.Conv1d(in_channels=in_channels, out_channels=HID_CHANNELS, kernel_size=7, stride=1,padding=3),
                                    nn.BatchNorm1d(HID_CHANNELS),
                                    nn.ReLU(),
                                    nn.Dropout(p=DROPOUT)
                                    )
        
        self.conv2 = nn.Sequential(
                                    nn.Conv1d(in_channels=HID_CHANNELS, out_channels=10, kernel_size=5, stride=1,padding=2),
                                    nn.BatchNorm1d(10),
                                    nn.ReLU(),
                                    nn.Dropout(p=DROPOUT)
                                    )
    
        
        self.dense = nn.Sequential(
                                    nn.Linear(in_features=10, out_features=1),
                                    )
        self.dropout = nn.Dropout(DROPOUT)
        # Dense out
        self.act = nn.ReLU()
        self.out = nn.Sigmoid()

    def forward(self, x):
        v = False

        x = x.permute(0,2,1)

        x = self.bn_input(x)

        # Convolutional modules
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.act(x)

        # Output MLP
        if v: print("Before permute", x.shape)
        x = x.permute(0,2,1)
        if v: print("After permute", x.shape)
        x = self.dense(x)

        # Output sigmoid
        out = self.out(x)

        return out

net = Simple1DCNN()
net.to(device)

################
### TRAINING ###
################

criterion= nn.BCELoss(reduction='none')
optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=LR_DECAY)

def weight_reset(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        m.reset_parameters()

net.apply(weight_reset)
losses=[]
training_plot=[]
test_plot=[]
auc_train_plot=[]
auc_test_plot=[]
mcc_train_plot=[]
mcc_test_plot=[]

last_score=np.inf
max_es_rounds = 5
es_rounds = max_es_rounds
best_epoch = 0

trainloader = torch.utils.data.DataLoader(training_set, batch_size=BATCH_SIZE)

for epoch in range(MAX_EPOCHS):  # loop over the dataset multiple times
    net.train()

    for i, data in enumerate(trainloader,0):

        # get the inputs; data is a list of [inputs, labels]

        inputs, labels, mask = data
        inputs = inputs.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        #scheduler.step()
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(torch.squeeze(outputs), labels)
        loss=loss*mask
        loss=torch.sum(loss)/torch.sum(mask)

        loss.backward()
        optimizer.step()

# print statistics
    with torch.no_grad():
        test_loss=0
        train_loss=0
        net.eval()
        inputs, labels, mask = training_set[:]

        outputs = net(inputs)

        loss = criterion(torch.squeeze(outputs), labels)
        loss=loss*mask
        loss=torch.sum(loss)/torch.sum(mask)
        training_plot.append(loss.cpu().numpy())
        auc_train_plot.append(roc_auc_score(labels.cpu()[mask.cpu()>0], outputs.cpu().squeeze()[mask.cpu()>0]))
        mcc_train_plot.append(mcc(labels.cpu()[mask.cpu()>0], outputs.cpu().squeeze()[mask.cpu()>0]>.1))


        inputs, labels, mask = test_set[:]
        outputs = net(inputs)
        loss = criterion(torch.squeeze(outputs), labels)
        loss=loss*mask
        loss=torch.sum(loss)/torch.sum(mask)
        test_plot.append(loss.cpu().numpy())
        #labels, outputs= get_labels_preds_and_posprob_without_padding( outputs.flatten(),labels.flatten() )
        auc_test_plot.append(roc_auc_score(labels.cpu()[mask.cpu()>0], outputs.cpu()[mask.cpu()>0].squeeze()))
        mcc_test_plot.append(mcc(labels.cpu()[mask.cpu()>0], outputs.cpu().squeeze()[mask.cpu()>0]>.1))

        inputs, labels, mask = val_set[:]

        outputs = net(inputs)
        fpr, tpr, _ = roc_curve(labels.cpu()[mask.cpu()>0], outputs.cpu()[mask.cpu()>0].squeeze())

        valloss = criterion(torch.squeeze(outputs), labels)
        valloss=valloss*mask
        valloss=torch.sum(valloss)/torch.sum(mask)

        print("Epoch {}, training loss {}, test loss {}, validation loss {}".format(epoch, training_plot[-1], test_plot[-1], valloss))
        print("Epoch {}, training AUC {}, test AUC {}".format(epoch, auc_train_plot[-1], auc_test_plot[-1]))
        print("Epoch {}, training MCC {}, test MCC {}".format(epoch, mcc_train_plot[-1], mcc_test_plot[-1]))


    if EARLY_STOPPING:
        if last_score > valloss:
            last_score = valloss
            best_epoch = epoch
            es_rounds = max_es_rounds
            best_model = copy.deepcopy(net)
            cnn_fpr, cnn_tpr = fpr, tpr
        else:
            if es_rounds > 0:
                es_rounds -=1
            else:
                print('EARLY-STOPPING !')
                print('Best epoch found: nº {}'.format(best_epoch))
                print('Exiting. . .')
                break

print("Finished training.")

