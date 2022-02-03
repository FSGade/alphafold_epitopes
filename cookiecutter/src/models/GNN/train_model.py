#!/usr/bin/env python3

print("Importing common model setup...")
import sys
sys.path.append("..")
from common_model_setup import *

SOLVED = False
print("Using {} structures for GNN".format("solved" if SOLVED else "AF2"))

device = "cuda" if torch.cuda.is_available() else "cpu"
print("The device in use is", device)

##############################################
### GNN NEEDS A COMPLETELY DIFFERENT SETUP ###
##############################################

from PS_model_for_finetuning import *
STATE_FILE = "../PS/ProteinSolver/e53-s1952148-d93703104.state"

EARLY_STOPPING = True

LR = 5e-4
MAX_EPOCHS = 200

if SOLVED:
    train_pt = '../../../data/processed/gnn/solved_train_data.pt'
    test_pt = '../../../data/processed/gnn/solved_test_data.pt'
    val_pt = '../../../data/processed/gnn/solved_val_data.pt'
else:
    train_pt = '../../../data/processed/gnn/af2_train_data.pt'
    test_pt = '../../../data/processed/gnn/af2_test_data.pt'
    val_pt = '../../../data/processed/gnn/af2_val_data.pt'

try:
    train_data = torch.load(train_pt, map_location=device)
    test_data = torch.load(test_pt, map_location=device)
    val_data = torch.load(val_pt, map_location=device)
except Exception as e:
    print(e)
    print("One of the GNN data files could not be loaded.")
    sys.exit(1)

num_features = 20
adj_input_size = 2
hidden_size = 128

#Define model
gnn = Net(
    x_input_size=num_features + 1,
    adj_input_size=adj_input_size,
    hidden_size=hidden_size,
    output_size=num_features
)

print("Model's state_dict:")
state_dict = torch.load(STATE_FILE, map_location=device)

state_dict['linear_pred_out.weight'] = torch.empty(1, 128)
state_dict['linear_pred_out.bias'] = torch.empty(1)
torch.nn.init.kaiming_uniform_(state_dict['linear_pred_out.weight'], a=math.sqrt(3))

fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(state_dict['linear_pred_out.weight'])
bound = 1 / math.sqrt(fan_in)
torch.nn.init.uniform_(state_dict['linear_pred_out.bias'], -bound, bound)

# Model weight shapes
for param_tensor in state_dict:
    print(param_tensor, "\t", state_dict[param_tensor].size())

# Set up network
gnn.load_state_dict(state_dict)
gnn.train()
gnn = gnn.to(device)


















################
### TRAINING ###
################

criterion= nn.BCELoss()
optimizer = optim.Adam(gnn.parameters(), lr=LR)

last_score = np.inf
max_es_rounds = 5
es_rounds = max_es_rounds
best_epoch = 0

auc_test = []

for epoch in range(MAX_EPOCHS):  # loop over the dataset multiple times
    gnn.train()

    losses = []
    true_vec = []
    pred_vec = []

    for i, data in enumerate(train_data):

        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        #inputs = inputs.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()
        #scheduler.step()     
        # forward + backward + optimize
        outputs = gnn(inputs.x, inputs.edge_index, inputs.edge_attr)

        #print(labels, outputs.shape)
        outputs = torch.flatten(outputs)
        loss = criterion(outputs, labels)
        losses.append(loss)

        true_vec.extend(list(labels.detach().cpu().numpy()))
        pred_vec.extend(list(outputs.detach().cpu().numpy()))

        loss.backward()
        optimizer.step()

    print("Epoch {}: Training loss: {}".format(epoch, sum(losses)/len(losses)))
    print("Epoch {}: Training AUC {}".format(epoch, roc_auc_score(true_vec, pred_vec)))

    gnn.eval()

    true_vec = []
    pred_vec = []
    
    for i, data in enumerate(test_data):
        inputs, labels = data
        outputs = gnn(inputs.x, inputs.edge_index, inputs.edge_attr)
        outputs = torch.flatten(outputs)

        true_vec.extend(list(labels.detach().cpu().numpy()))
        pred_vec.extend(list(outputs.detach().cpu().numpy()))

    fpr, tpr, _ = roc_curve(true_vec, pred_vec)
    auc_test.append(roc_auc_score(true_vec, pred_vec))
    print("Epoch {}: Test AUC {}".format(epoch, auc_test[epoch]))

    true_vec = []
    pred_vec = []
    val_loss = 0

    for i, data in enumerate(val_data):
        inputs, labels = data
        outputs = gnn(inputs.x, inputs.edge_index, inputs.edge_attr)
        outputs = torch.flatten(outputs)
        val_loss += criterion(outputs, labels).detach().cpu().numpy()

        true_vec.extend(list(labels.detach().cpu().numpy()))
        pred_vec.extend(list(outputs.detach().cpu().numpy()))

    print("Epoch {}: Validation AUC {}".format(epoch, roc_auc_score(true_vec, pred_vec)))
    print("Validation loss {}".format(val_loss))
    print()

    if EARLY_STOPPING:
        if last_score > val_loss:
            last_score = val_loss
            best_epoch = epoch
            es_rounds = max_es_rounds
            best_model = copy.deepcopy(gnn)
            gnn_fpr, gnn_tpr = fpr, tpr
        else:
            if es_rounds > 0:
                es_rounds -=1
            else:
                print('EARLY-STOPPING !')
                print('Best epoch found: nº {}'.format(best_epoch))
                print('Exiting. . .')
                break

