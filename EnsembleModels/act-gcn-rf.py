import torch
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F
import torch_geometric.transforms as T

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
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

def drawROC(test_y,y_pred):
    fpr, tpr, thersholds = roc_curve(test_y,y_pred, pos_label=1)
    roc_auc = auc(fpr, tpr)
    return roc_auc


# read data
df_actor = pd.read_csv('trainData/df_actor.csv')
df_edges = pd.read_csv('trainData/df_edges.csv')

try:
    df_actor.drop('Unnamed: 0',axis=1,inplace=True)
except:
    pass

# 尝试剔除不重要features
# df_actor.drop(['blocks_btwn_input_txs_median',
#  'first_sent_block',
#  'first_received_block',
#  'fees_mean',
#  'first_block_appeared_in',
#  'fees_total',
#  'fees_min',
#  'fees_as_share_min',
#  'blocks_btwn_txs_max',
#  'fees_max'],axis=1,inplace=True)

# 归一化
for column in df_actor.columns[3:]:
    feature = np.array(df_actor[column]).reshape(-1,1)
    scaler = MinMaxScaler()
    scaler.fit(feature)
    feature_scaled = scaler.transform(feature)
    df_actor[column] = feature_scaled.reshape(1,-1)[0]





# 数据处理
df_actor['class'] = df_actor['class'].replace(2,0)
index_range = df_actor[df_actor['Time step'].between(1, 34)].index.tolist()


train = df_actor.iloc[index_range]
test =  df_actor.drop(index=index_range)
print('train shape: ',train.shape[0],'\ntest shape: ',test.shape[0])

y_train = train['class']
X_train = train.drop(['class','address','Time step'],axis=1)
y_test = test['class']
X_test = test.drop(['class','address','Time step'],axis=1)


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

        return x

# evaluation 函数
def eval_classifier(y_pred,y_test):
    roc_auc = drawROC(y_test,y_pred)
    f1 = f1_score(y_test , y_pred, average='binary', pos_label=1)
    recall = recall_score(y_test , y_pred, average='binary', pos_label=1)
    precision= precision_score(y_test , y_pred, average='binary', pos_label=1)
    acc = accuracy_score(y_test , y_pred)

    print(f'Accuracy: {acc :.4f}')
    print(f'Recall: {recall :.4f}')
    print(f'Precision: {precision :.4f}')
    print(f'F1: {f1 :.4f}')
    print(f'AUC_ROC: {roc_auc :.4f}')
    


def train(epoches,weight,hidden_layer,data,output_channels):
    # 设定输出评估权重
    ary = np.ones(output_channels-1)
    ary = np.append(ary,[weight])
    label_weight = torch.tensor(ary,dtype=torch.float)
    del ary
  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gcn_model = GNN(num_features=data.x.size(dim=1), hidden_layer=hidden_layer, output_channels=output_channels).to(device)
    data = data.to(device)
    label_weight = label_weight.to(device)
    optimizer = torch.optim.Adam(gcn_model.parameters(), lr=0.001)

    for epoch in range(epoches):
        optimizer.zero_grad()
        out = gcn_model(data)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask],weight=label_weight)
        loss.backward()
        optimizer.step()
        if epoch %100 == 0:
            print(f"epoch:{epoch+1}, loss:{loss.item()}")
    return gcn_model
    

for h_l in [100]:
    for weight in [3]:
        for output_channels in [2,12,24,36,57]:       
                # Generate convoluted features using trained GCN model
                convoluted_features = train(epoches=800,weight=weight,hidden_layer=h_l,data=data,output_channels=output_channels)#gcn_model(data.x, edge_index)
                convoluted_features = convoluted_features(data)
                train_convoluted = convoluted_features[train_mask].detach()

                rf_conv = RandomForestClassifier(n_estimators=50,random_state=42)
                rf_conv.fit(train_convoluted, data.y[train_mask])

                # Train Random Forest models
                rf_no_conv = RandomForestClassifier(n_estimators=50,random_state=42)
                rf_no_conv.fit(X_train, y_train)

                # Evaluate both models
                pred_conv = rf_conv.predict(convoluted_features[test_mask].detach().numpy())
                pred_no_conv = rf_no_conv.predict(X_test)

                print('800 epoches and output channel: %d '%(output_channels))
                print('Resutl With GCN')
                eval_classifier(pred_conv,y_test)

                print('\nResutl With only RF')
                eval_classifier(pred_no_conv,y_test)
                del rf_conv, rf_no_conv, convoluted_features






