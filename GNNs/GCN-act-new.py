import torch
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch_geometric.transforms as T

from sklearn.preprocessing import MinMaxScaler
from torch_geometric.nn import GCNConv,GATConv,SAGEConv
from torch_geometric.data import Data
# from torch_geometric.transforms import RandomNodeSplit
from sklearn.metrics import f1_score, recall_score, precision_score,confusion_matrix,roc_curve,accuracy_score,auc


import pandas as pd

def load_node_csv(df, index_col,y=None, encoders=None,**kwargs):
    mapping = {index: i for i, index in enumerate(df[index_col].unique())}
    x = df.drop([y,index_col,'Time step'],axis=1)
    x = torch.tensor(x.values.astype(np.float32))
    y = torch.tensor(df[y].values.astype(np.int64))#.int()
    # if encoders is not None:
    #     xs = [encoder(df[col]) for col, encoder in encoders.items()]
    #     x = torch.cat(xs, dim=-1)

    return x, y, mapping


def load_edge_csv(df, src_index_col, src_mapping, dst_index_col, dst_mapping,
                  encoders=None, **kwargs):
    src = [src_mapping[index] for index in df[src_index_col]]
    dst = [dst_mapping[index] for index in df[dst_index_col]]
    edge_index = torch.tensor([src, dst])

    edge_attr = None
    if encoders is not None:
        edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
        edge_attr = torch.cat(edge_attrs, dim=-1)

    return edge_index, edge_attr


df_actor = pd.read_csv('trainData/df_actor.csv')
df_edges = pd.read_csv('trainData/df_edges.csv')


df_actor['class'] = df_actor['class'].replace(2,0)
df_actor= df_actor.sort_values(by=['Time step'])
index_range = df_actor[df_actor['Time step'].between(1, 34)].index.tolist()








for column in df_actor.columns[3:]:
    feature = np.array(df_actor[column]).reshape(-1,1)
    scaler = MinMaxScaler()
    scaler.fit(feature)
    feature_scaled = scaler.transform(feature)
    df_actor[column] = feature_scaled.reshape(1,-1)[0]

try:
    df_actor.drop('Unnamed: 0',axis=1,inplace=True)
#    df_edges.drop('index',axis=1,inplace=True)
except:
    pass




# read data to graph
df_actor_x, df_actor_y, df_actor_mapping = load_node_csv(df_actor,index_col='address',y='class')

edge_index,_ = load_edge_csv(df_edges, 'input_address',  df_actor_mapping , 'output_address',  df_actor_mapping ,
                  encoders=None)

data = Data(x=df_actor_x,y=df_actor_y,edge_index=edge_index)


train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
train_mask[0:len(index_range)] = True

test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
test_mask[len(index_range):] = True

data.train_mask = train_mask
data.test_mask = test_mask


data.train_mask = train_mask
data.test_mask = test_mask

# split = T.RandomNodeSplit(num_test=0.2)
# data = split(data)



# GNN model
class GNN(torch.nn.Module):
    def __init__(self, num_features, hidden_layer, output_channels):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(num_features,hidden_layer )
        self.conv2 = GCNConv(hidden_layer, output_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


def eval_node_classifier(model, graph, mask):
    model.eval()
    pred = model(graph).argmax(dim=1)

    
    y_pred = pred[mask]
    test_y = graph.y[mask]
    # print('y_pred:',y_pred)
    # print('\ntest_y',test_y)
    roc_auc = drawROC(test_y,y_pred)
    f1 = f1_score(test_y , y_pred, average='binary', pos_label=1)
    recall = recall_score(test_y , y_pred, average='binary', pos_label=1)
    precision= precision_score(test_y , y_pred, average='binary', pos_label=1)
    acc = accuracy_score(test_y , y_pred)
    
    # test_acc,test_recall,test_f1,test_precision = eval_node_classifier(model, data, data.test_mask)
    print(f'Accuracy: {acc :.4f}')
    print(f'Recall: {recall :.4f}')
    print(f'Precision: {precision :.4f}')
    print(f'F1: {f1 :.4f}')
    print(f'AUC_ROC: {roc_auc :.4f}')
    
    

    # return acc,recall,f1,precision

def drawROC(test_y,y_pred):
    fpr, tpr, thersholds = roc_curve(test_y,y_pred, pos_label=1)
    roc_auc = auc(fpr, tpr)
    return roc_auc
 
    # plt.plot(fpr, tpr, 'k--', label='ROC (area = {0:.2f})'.format(roc_auc), lw=2)
    
    # plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
    # plt.ylim([-0.05, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')  
    # plt.title('ROC Curve')
    # plt.legend(loc="lower right")
    # plt.show()
    # plt.savefig('roc-actor with val old weight 3 times features and 2000 epoche.png')





# torch.save(model, './model/df_actor-2000.pth')


def train(epoches,weight,hidden_layer,data,output_channels):
    ary = np.ones(output_channels-1)
    ary = np.append(ary,[weight])
    label_weight = torch.tensor(ary,dtype=torch.float)
    del ary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNN(num_features=df_actor.shape[1]-3, hidden_layer=hidden_layer, output_channels=output_channels).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epoches):
        optimizer.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask],weight=label_weight)
        loss.backward()
        optimizer.step()
        if epoch %100 == 0:
            print(f"epoch:{epoch+1}, loss:{loss.item()}")

    print('Actor model with %d weight, %d hidden layer, and%d+1 epoche'%(weight,hidden_layer,epoch))
    eval_node_classifier(model, data, data.test_mask)
    print('*' * 25)

for weight in [5,8,10]:
    for hl in [200,300,360]:
        train(epoches=1000,weight=weight,hidden_layer=hl,data=data,output_channels=2)

    # torch.save(model.state_dict(), 'GNN-actor-2.pth')



