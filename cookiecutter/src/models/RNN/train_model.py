#!/usr/bin/env python3

print("Importing common model setup...")
import sys
sys.path.append("..")
from common_model_setup import *

SOLVED = False
print("Using {} structures for RNN".format("solved" if SOLVED else "AF2"))

device = "cuda" if torch.cuda.is_available() else "cpu"
print("The device in use is", device)

BATCH_SIZE = 8
MAX_EPOCHS = 200
LR = 1e-3
LR_DECAY = 1e-5

EARLY_STOPPING = True

training_set, val_set, test_set = create_sequential_datasets(
        *create_embedding_dicts(SOLVED), device
        )

class SimpleRNN(torch.nn.Module):
    def __init__(self, input_size):
        super(SimpleRNN, self).__init__()
        #self.bn_input = nn.BatchNorm1d(927)

        self.n_layers = 1
        self.hidden_dim = 32

        #self.rnn = nn.RNN(input_size, self.hidden_dim, self.n_layers, batch_first=True)
        self.rnn = nn.RNN(input_size, self.hidden_dim, self.n_layers, batch_first=True, bidirectional=True)
        
    
        self.dense = nn.Sequential(
                                    nn.Linear(in_features=2*self.hidden_dim, out_features=1)
#                                    nn.Linear(in_features=2*self.hidden_dim, out_features=self.hidden_dim),
#                                    nn.ReLU(),
#                                    nn.Linear(in_features=self.hidden_dim, out_features=int(self.hidden_dim/2)),
#                                    nn.ReLU(),
#                                    nn.Linear(in_features=int(self.hidden_dim/2), out_features=int(self.hidden_dim/4)), 
#                                    nn.ReLU(),
#                                    nn.Linear(in_features=int(self.hidden_dim/4), out_features=1),
                                    )
        # Dense out
        #self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.out = nn.Sigmoid()

    def forward(self, x):
        #x = self.bn_input(x)
    
        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(4)

        # Convolutional modules
        x, hidden = self.rnn(x)
        #x = self.act(x)
        x = self.dropout(x)

        # Output MLP
        x = self.dense(x)

        # Output sigmoid
        out = self.out(x)

        return out, hidden

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden

net = SimpleRNN(128)
net.to(device)

trainloader = torch.utils.data.DataLoader(training_set, batch_size=BATCH_SIZE)

################
### TRAINING ###
################

print("Starting training...")

train_X, train_y, train_mask = training_set[:]
test_X, test_y, test_mask = test_set[:]

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
        outputs, hidden = net(inputs)
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

        outputs, hidden = net(inputs) 

        loss = criterion(torch.squeeze(outputs), labels)
        loss=loss*mask
        loss=torch.sum(loss)/torch.sum(mask)
        training_plot.append(loss.cpu().numpy())
        auc_train_plot.append(roc_auc_score(labels.cpu()[mask.cpu()>0], outputs.cpu().squeeze()[mask.cpu()>0]))
        mcc_train_plot.append(mcc(labels.cpu()[mask.cpu()>0], outputs.cpu().squeeze()[mask.cpu()>0]>.1))
 

        inputs, labels, mask = test_set[:]
        outputs, hidden = net(inputs)
        loss = criterion(torch.squeeze(outputs), labels)
        loss=loss*mask
        loss=torch.sum(loss)/torch.sum(mask)       
        test_plot.append(loss.cpu().numpy())
        fpr, tpr, _ = roc_curve(labels.cpu()[mask.cpu()>0], outputs.cpu()[mask.cpu()>0].squeeze())
        #labels, outputs= get_labels_preds_and_posprob_without_padding( outputs.flatten(),labels.flatten() )
        auc_test_plot.append(roc_auc_score(labels.cpu()[mask.cpu()>0], outputs.cpu()[mask.cpu()>0].squeeze()))
        mcc_test_plot.append(mcc(labels.cpu()[mask.cpu()>0], outputs.cpu().squeeze()[mask.cpu()>0]>.1))
        
        inputs, labels, mask = val_set[:]

        outputs, hidden = net(inputs) 

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
            rnn_fpr, rnn_tpr = fpr, tpr
        else:
            if es_rounds > 0:
                es_rounds -=1
            else:
                print('EARLY-STOPPING !')
                print('Best epoch found: nº {}'.format(best_epoch))
                print('Exiting. . .')
                break

print("Finished training.")

