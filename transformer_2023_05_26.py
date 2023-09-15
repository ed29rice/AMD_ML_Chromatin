import os
os.system('pip install -q glob2==0.7 requests pytest-shutil==1.7.0  pyBigWig==0.3.18 urllib3==1.26.14 tqdm==4.64.1 joblib==1.2.0 ipywidgets==8.0.4 biopython')
os.system('pip install umap-learn')
os.system('pip install pydca --no-deps')
os.system('pip install -i https://test.pypi.org/pypi/ --extra-index-url https://pypi.org/simple PyMEGABASE==1.0.13 --no-deps')

import PyMEGABASE as PYMB
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from tempfile import TemporaryDirectory
import time
import seaborn as sns
from typing import Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import functional as F

#Initialize PyMEGABASE
pymb=PYMB.PyMEGABASE(cell_line='A549', assembly='GRCh38', organism='human',
                    signal_type='signal p-value',ref_cell_line_path='./GM12878_hg19_test',
                    histones=True,tf=False,small_rna=False,total_rna=False,n_states=10,res=50)
pymb.training_set_up(filter=True)

test_cell=[]
for chr in range(1,23):
    test_cell.append(pymb.test_set(chr=chr,silent=True))
test_cell=np.concatenate(test_cell,axis=1)

tmp_all_matrix=pymb.get_tmatrix(range(1,23))
nfeatures=len(np.loadtxt(pymb.cell_line_path+'/unique_exp.txt',dtype=str))
print(nfeatures)

n_neigbors=5
#Populate training and validation set
tmp=[]
for l in range(n_neigbors,len(tmp_all_matrix[0])-n_neigbors):
    tmp.append(np.insert(np.concatenate(tmp_all_matrix[nfeatures*2+1:nfeatures*3+1,l-n_neigbors:l+n_neigbors+1]),0,tmp_all_matrix[0,l]))
all_matrix=np.array(tmp).T

#Populate prediction set
tmp=[]
for l in range(n_neigbors,len(test_cell[0])-n_neigbors):
    tmp.append(np.insert(np.concatenate(test_cell[nfeatures*2:nfeatures*3,l-n_neigbors:l+n_neigbors+1]),0,1))
all_test_cell=np.array(tmp).T

nfeatures=(2*n_neigbors+1)*nfeatures
del tmp
print(nfeatures,all_test_cell.shape,all_matrix.shape)

tidx=np.random.choice(np.linspace(0,len(all_matrix[0])-1,len(all_matrix[0])).astype(int),size=int(0.8*len(all_matrix[0])),replace=False)

ttidx=np.zeros(len(all_matrix[0])).astype(bool)
ttidx[tidx]=1
ttidx

tmatrix=all_matrix[:,ttidx]
vmatrix=all_matrix[:,~ttidx][:,::2]
testmatrix=all_matrix[:,~ttidx][:,1::2]
tmatrix.shape,vmatrix.shape,testmatrix.shape,all_matrix.shape

#Preprocess the downloaded data for tranining
training_averages=tmatrix.T
n_exp=int(training_averages[:,1:].shape[1]/(2*n_neigbors+1))
train_X=training_averages[:,1:].reshape(len(training_averages),(2*n_neigbors+1),n_exp)
test_set=testmatrix.T
test_X=test_set[:,1:].reshape(len(test_set),(2*n_neigbors+1),n_exp)

plt.figure()
plt.matshow(train_X[1000])
plt.figure()
plt.matshow(test_X[2000])

class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, features: int, ostates: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.d_model = d_model
        self.encoder = nn.Embedding(ntoken, d_model)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.fl = nn.Flatten()
        self.l2 = nn.Linear(features*d_model,ostates)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.l2.bias.data.zero_()
        self.l2.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        #src = self.pos_encoder(src)
        output_tf = self.transformer_encoder(src, src_mask)
        output = self.fl(output_tf)
        output = self.l2(output)
        return  F.log_softmax(output, dim=-1), output_tf

training_averages=tmatrix.T-1
validation_set=vmatrix.T-1
test_set=testmatrix.T-1

train_data=torch.tensor(training_averages.astype(int))
val_data=torch.tensor(validation_set.astype(int))
test_data=torch.tensor(test_set.astype(int))
n_exp=int(training_averages[:,1:].shape[1]/5)
print(train_data.size(),val_data.size(),test_data.size())

nbatches=200
bptt = len(training_averages)//nbatches
print('Size of batch:',bptt)
def get_batch(source: Tensor, i: int) -> Tuple[Tensor, Tensor]:
    data = source[i*bptt:(i+1)*bptt,1:]
    target = source[i*bptt:(i+1)*bptt,0]
    return data.to(device), target.to(device)

from torch.utils.tensorboard import SummaryWriter
def train(model: nn.Module, writerTB: SummaryWriter) -> None:
    model.train()  # turn on train mode
    total_loss = 0.
    start_time = time.time()
    src_mask = None

    num_batches = len(train_data) // bptt
    bs=list(range(nbatches))
    np.random.shuffle(bs)
    for batch in bs:
        i=batch
        data, targets = get_batch(train_data, i)
        seq_len = data.size(0)
        output, output_tf = model(data, src_mask)
        loss = criterion(output.view(-1, 5), targets)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss = total_loss + loss.item()
    writerTB.add_scalar('loss_per_epoch', total_loss, epoch)
    return total_loss

def evaluate(model: nn.Module, eval_data: Tensor, writerTB=None, type='test') -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    src_mask = None
    nbatches_eval=len(eval_data)//bptt
    with torch.no_grad():
        for batch in range(nbatches_eval):
            data, targets = get_batch(eval_data, batch)
            output, output_tf = model(data, src_mask)
            total_loss = total_loss + criterion(output.view(-1, 5), targets).item()
    if type=='val':
        writerTB.add_scalar('vloss_per_epoch', total_loss, epoch)
    return total_loss

"""# Training based on training loss"""

ntokens = int(training_averages.max())+1  # size of vocabulary
emsize = 32 # embedding dimension
d_hid = 32 # dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 1 # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2  # number of heads in nn.MultiheadAttention
dropout = 0.5  # dropout probability
ostates = 5 # Output states
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, nfeatures, ostates, dropout).to(device)
criterion = nn.CrossEntropyLoss()
print(model.modules)
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print('Number of params:',params)

best_val_loss = float('inf')
best_train_loss = float('inf')
epochs = 50
nepochs = 50

lr = 5  # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

with TemporaryDirectory() as tempdir:
    writer = SummaryWriter()
    bsteps=0
    for epoch in range(1, epochs + 1):
        if epoch%nepochs==1:
            print('Reseting lr')
            optimizer = torch.optim.SGD(model.parameters(), lr=lr,weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5)
            train_loss=train(model, writer)
        lr_e = scheduler.get_last_lr()[0]
        epoch_start_time = time.time()
        train_loss=train(model, writer)
        elapsed = time.time() - epoch_start_time
        val_loss = evaluate(model, val_data, writer, type='val')
        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
            f'valid loss {val_loss:5.2f} | train loss {train_loss:5.2f} | lr {lr_e:2.5f} | best val loss: {best_val_loss:5.2f}')
        print('-' * 89)

        if train_loss < best_train_loss:
            print('Found better train_loss')
            best_train_loss = train_loss
            torch.save(model.state_dict(), './best_train_model_params.pt')
            bsteps=bsteps+2
        if val_loss < best_val_loss:
            print('Found better val_loss')
            best_val_loss = val_loss
            torch.save(model.state_dict(), './best_val_model_params.pt')
        else:
            bsteps=bsteps+1

        if bsteps>5:
            print('Has not found better train_loss in 5 epochs, reducing lr')
            scheduler.step()
            print(scheduler.get_last_lr()[0])
            bsteps=0
torch.save(model.state_dict(), './last_model_params.pt')
writer.close()
print('=' * 89)
print(f'| End of training | best val loss {best_val_loss:5.2f} | ')

test_loss = evaluate(model, test_data)
print('=' * 89)
print(f'| End of training | test loss {test_loss:5.2f} | ')
print('=' * 89)

# Commented out IPython magic to ensure Python compatibility.
# Load the TensorBoard notebook extension
# %load_ext tensorboard
# %tensorboard --logdir=runs

ll={}
k=0
plt.figure()
for p in ['best_val']:
    model.load_state_dict(torch.load('./'+p+'_model_params.pt'))
    nbatches_eval=len(train_data)//bptt
    l=[]
    lt=[]
    with torch.no_grad():
        for batch in range(nbatches_eval):
            data, targets = get_batch(train_data, batch)
            l.append(model(data,None)[0].argmax(dim=1).cpu())
            lt.append(targets.cpu())
    l=np.concatenate(l)
    lt=np.concatenate(lt)
    print('train:',np.round(np.sum(l==lt)/len(l),3))
    plt.bar([0+k],[np.sum(l==lt)/len(l)],width=0.8,facecolor='k')
    nbatches_eval=len(val_data)//bptt
    l=[]
    lt=[]
    with torch.no_grad():
        for batch in range(nbatches_eval):
            data, targets = get_batch(val_data, batch)
            l.append(model(data,None)[0].argmax(dim=1).cpu())
            lt.append(targets.cpu())
    l=np.concatenate(l)
    lt=np.concatenate(lt)
    print('val:',np.round(np.sum(l==lt)/len(l),3))
    plt.bar([1+k],[np.sum(l==lt)/len(l)],width=0.8,facecolor='b')
    nbatches_eval=len(test_data)//bptt
    l=[]
    lt=[]
    with torch.no_grad():
        for batch in range(nbatches_eval):
            data, targets = get_batch(test_data, batch)
            l.append(model(data,None)[0].argmax(dim=1).cpu())
            lt.append(targets.cpu())
    l=np.concatenate(l)
    lt=np.concatenate(lt)
    print('test:',np.round(np.sum(l==lt)/len(l),3))
    plt.bar([2+k],[np.sum(l==lt)/len(l)],width=0.8,facecolor='r')
    k=k+0.5
plt.ylim([0.4,0.7])
plt.xticks([0,1,2],['train','val','test'])

conf_matrix_P=np.zeros((5,5))
conf_matrix_M=np.zeros((5,5))
int_types_Or=lt
types_pyME=l
for i in range(5):
    idx1=(int_types_Or==i)
    subav=int_types_Or[idx1]
    subprd_P=types_pyME[idx1]
    for k in range(len(subprd_P)):
        for j in range(5):
            conf_matrix_P[i,j]+=(subprd_P[k]==j)
    conf_matrix_P[i,:]=np.round(conf_matrix_P[i,:]/np.sum(idx1),3)

print('Confusion Matrix')
fig, axs = plt.subplots(1, 1,figsize=(10,8))
sns.heatmap(conf_matrix_P,annot=conf_matrix_P,cmap='binary',ax=axs)
plt.yticks([0.5,1.5,2.5,3.5,4.5],['A1','A2','B1','B2','B3'],fontsize=20)
plt.xticks([0.5,1.5,2.5,3.5,4.5],['A1','A2','B1','B2','B3'],fontsize=20)
plt.show()

int_types_Or=lt;types_pyME=l
idx=(int_types_Or!=6)
AB=np.copy(int_types_Or[idx])
AB[np.where((int_types_Or[idx]==0) | (int_types_Or[idx]==1))]=1
AB[np.where((int_types_Or[idx]==2) | (int_types_Or[idx]==3) | (int_types_Or[idx]==4))]=0
preAB=np.copy(types_pyME[idx])
preAB[np.where((types_pyME[idx]==0) | (types_pyME[idx]==1))]=1
preAB[np.where((types_pyME[idx]==2) | (types_pyME[idx]==3) | (types_pyME[idx]==4))]=0

conf_matrix=np.zeros((2,2))
for i in range(2):
    idx1=(AB==i)
    subav=AB[idx1]
    subprd=preAB[idx1]
    for k in range(len(subprd)):
        for j in range(2):
            conf_matrix[i,j]+=(subprd[k]==j)
    conf_matrix[i,:]=np.round(conf_matrix[i,:]/np.sum(idx1),3)

print(conf_matrix)
print('AB Confusion Matrix')
sns.heatmap(conf_matrix,annot=conf_matrix,cmap='binary')
print(np.sum(preAB==AB)/len(preAB))

def get_test_set(cellname,n_neigbors,chrms=range(1,23)):
    #Initialize PyMEGABASE
    _pymb=PYMB.PyMEGABASE(cell_line=cellname, assembly='GRCh38', organism='human',
    signal_type='signal p-value',ref_cell_line_path='./GM12878_hg19_test',
    histones=True,tf=False,small_rna=False,total_rna=False,n_states=10,res=50)
    print('looking for data in:',_pymb.cell_line_path)
    test_cell=[]
    for chr in chrms:
        test_cell.append(_pymb.test_set(chr=chr,silent=True))
    test_cell=np.concatenate(test_cell,axis=1)
    #Populate prediction set
    tmp=[]
    nfeatures=len(np.loadtxt(_pymb.cell_line_path+'/unique_exp.txt',dtype=str))
    for l in range(n_neigbors,len(test_cell[0])-n_neigbors):
        tmp.append(np.insert(np.concatenate(test_cell[nfeatures*2:nfeatures*3,l-n_neigbors:l+n_neigbors+1]),0,1))
    testmatrix=np.array(tmp).T
    nfeatures=(2*n_neigbors+1)*nfeatures
    test_set=testmatrix.T-1
    test_data=torch.tensor(test_set.astype(int))
    return test_data

subc={}
for name in ['K562','IMR-90','A549','HepG2']:
    ts=get_test_set(name,n_neigbors,chrms=[1])
    nbatches_eval=len(ts)//bptt
    l=[]
    with torch.no_grad():
        for batch in range(nbatches_eval):
            data, targets = get_batch(ts, batch)
            l.append(model(data,None)[0].argmax(dim=1).cpu())
    subc[name]=np.concatenate(l)
    u,c=np.unique(subc[name],return_counts=True)
    print(name,u,c,'A:',np.sum(c[:2]),'- B:',np.sum(c[2:]))

for name in ['K562','IMR-90','A549','HepG2']:
    plt.figure(figsize=(5,5))
    plt.bar(['A','B'],[np.sum(subc[name]<2)/len(subc[name]),np.sum(subc[name]>=2)/len(subc[name])])
    plt.yticks(fontsize=20)
    plt.title(name,fontsize=20)
    plt.xticks(fontsize=20)
    plt.show()

for name in ['K562','IMR-90','A549','HepG2']:
    plt.figure(figsize=(5,5))
    u,c=np.unique(subc[name],return_counts=True)
    plt.bar(u,c/len(subc[name]))
    plt.title(name,fontsize=20)
    plt.xticks([0,1,2,3,4],['A1','A2','B1','B2','B3'],fontsize=20)
    plt.show()

chrbound=[1,5000]
for name in ['K562','IMR-90','A549','HepG2']:
    plt.figure(figsize=(10,5))
    x=np.linspace(1,len(subc[name]),len(subc[name]))
    for s in [0,1,2,3,4]:
            plt.plot(x[subc[name]==s],subc[name][subc[name]==s],'|',markersize=5)
    plt.yticks([0,1,2,3,4],['A1','A2','B1','B2','B3'],fontsize=20)
    plt.xlim(chrbound)
    plt.title(name,fontsize=20)
    plt.xticks(fontsize=20)
    plt.show()

PYMB_sub.shape

PYMB_sub

name='HepG2'
colors=['#fb0200','#f7b12e','#0705f0','#06bdbc','#09820a']
PYMB_sub=np.loadtxt('./chr1_subcompartments.txt',dtype=str)
PYMB_sub=PYMB_sub[:len(subc[name]),1]
PYMB_sub
TYPE_TO_INT = {'A1':0,'A2':1,'B1':2,'B2':3,'B3':4,'B4':5,'NA':6}
for i in range(len(PYMB_sub)):
    PYMB_sub[i]=TYPE_TO_INT[PYMB_sub[i]]
PYMB_sub=PYMB_sub.astype(int)
plt.figure(figsize=(20,5))
x=np.linspace(1,len(subc[name]),len(subc[name]))
for s in [0,1,2,3,4]:
    plt.plot(x[subc[name]==s],subc[name][subc[name]==s],'|',markersize=5,c=colors[s])
    plt.plot(x[PYMB_sub==s],PYMB_sub[PYMB_sub==s]+0.1,'|',markersize=5,c=colors[s])
plt.yticks([0,1,2,3,4],['A1','A2','B1','B2','B3'],fontsize=20)
plt.xlim(chrbound)
plt.title(name,fontsize=20)
plt.xticks(fontsize=20)
plt.show()

from sklearn.metrics import confusion_matrix
plt.figure()
plt.matshow(np.round(confusion_matrix(subc[name],PYMB_sub,normalize='pred'),2),cmap='binary')
plt.xlim([-0.5,4.5])
plt.ylim([-0.5,4.5])
plt.colorbar()

