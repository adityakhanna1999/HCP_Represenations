import torch
from torch import nn
import numpy as np
import pandas as pd
from mode import Model
import pickle



# Parameters
file="data.csv"
channel_embedDimension=10
brand_embedDimension=10
HCP_embedDimension=20
disposition_embedDimension=5
hidden_dim=12
batch_size=3
n_epochs = 10
lr=0.01
concat_size=channel_embedDimension+brand_embedDimension+HCP_embedDimension+disposition_embedDimension

def convert_to_indexing(a,indexs):
	x=len(indexs)
	for i in range(len(a)):
		for j in range(len(a[i])):
			if a[i][j]== '<EOS>':
				a[i][j]=0
			elif a[i][j]== '<PAD>':
				a[i][j]=x+1
			else : 
				a[i][j] = indexs[a[i][j]]
	return a 

df=pd.read_csv(file)
print(len(df))

# Get input Tables for all the attributes
HCP_feats=[]
brand_feats=[]
channel_feats=[]
disposition_feats=[]
Y=[]
max_seq_length=0
grouped_df=df.groupby(['zs_id','BRAND'])
print(len(grouped_df))
for key, item in grouped_df:
	df1=grouped_df.get_group(key)
	brand=df1['BRAND']
	hcp=df1['zs_id']
	channel=df1['Z_CHN_CD']
	disposition=df1['STANDARD_DISPOSITION']
	Nmonth=df1['Nmonth']
	list_brand=[]
	list_hcp=[]
	list_channel=[]
	list_disposition=[]
	for i in range(len(brand)):
		if (Nmonth.iloc[i] <= 20):
			list_brand.append(brand.iloc[i])
			list_hcp.append(hcp.iloc[i])
			list_channel.append(channel.iloc[i])
			list_disposition.append(disposition.iloc[i])


	list_brand.append('<EOS>')
	list_hcp.append('<EOS>')
	list_channel.append('<EOS>')
	list_disposition.append('<EOS>')

	HCP_feats.append(list_hcp)
	brand_feats.append(list_brand)
	channel_feats.append(list_channel)
	disposition_feats.append(list_disposition)

# Find max_seq_length for Padding and Indexing
for i in range(len(HCP_feats)):
	if(len(HCP_feats[i])>max_seq_length):
		max_seq_length=len(HCP_feats[i])
print(max_seq_length)


# Padd the Sequences
for i in range(len(HCP_feats)):
	x=len(HCP_feats[i])
	while(x<max_seq_length):
		HCP_feats[i].append('<PAD>')
		channel_feats[i].append('<PAD>')
		disposition_feats[i].append('<PAD>')
		brand_feats[i].append('<PAD>')
		x=x+1


# One hot encode the Indexes 
brand_index=pd.unique(df['BRAND'])
brand_vocabsize=len(brand_index)+2

hcp_index=pd.unique(df['zs_id'])
HCP_vocabsize=len(hcp_index)+2

channel_index=pd.unique(df['Z_CHN_CD'])
channel_vocabsize=len(channel_index)+2 

disposition_index=pd.unique(df['STANDARD_DISPOSITION'])
disposition_vocabsize=len(disposition_index)+2

int2char = dict(enumerate(brand_index))
brand_index = {char: ind+1 for ind, char in int2char.items()}

int2char = dict(enumerate(hcp_index))
hcp_index = {char: ind+1 for ind, char in int2char.items()}

int2char = dict(enumerate(channel_index))
channel_index = {char: ind+1 for ind, char in int2char.items()}

int2char = dict(enumerate(disposition_index))
disposition_index = {char: ind+1 for ind, char in int2char.items()}

HCP_feats = convert_to_indexing(HCP_feats,hcp_index)
brand_feats = convert_to_indexing(brand_feats,brand_index)
channel_feats = convert_to_indexing(channel_feats,channel_index)
disposition_feats = convert_to_indexing(disposition_feats,disposition_index)


# for i in range(len(HCP_feats)):
# 	print(HCP_feats[i],"XXX",brand_feats[i],"XXXX",channel_feats[i],"XXXXX",disposition_feats[i])

##################################
# Getting Output Labels 

labels=['DPH','DPL','DRT','PI','PR']
for key, item in grouped_df:
	df1=grouped_df.get_group(key)
	Nmonth=df1['Nmonth']
	channel_1=df1['Z_CHN_CD']
	brand=df1['BRAND']
	dict_channel={}
	for i in labels:
		dict_channel[i]=0
	total=0
	for i in range(len(brand)):
		if (Nmonth.iloc[i] > 20 and Nmonth.iloc[i] <= 24):
			dict_channel[channel_1.iloc[i]]+=1
			total+=1
	temp=[]
	for i in dict_channel.keys():
		if total!=0:
			dict_channel[i]=dict_channel[i]/total
		temp.append(dict_channel[i])
	Y.append(temp)
print(Y)

#### Training loop 
Brand_feats=torch.tensor(brand_feats,dtype=torch.long)
Channel_feats=torch.tensor(channel_feats,dtype=torch.long)
HHCP_feats=torch.tensor(HCP_feats,dtype=torch.long)
Disposition_feats=torch.tensor(disposition_feats,dtype=torch.long)

Y=torch.tensor(Y)

size=Y.size()[0]
batches=int(size/batch_size)
print(batches)
model= Model(channel_vocabsize=channel_vocabsize, channel_embedDimension=channel_embedDimension, brand_vocabsize=brand_vocabsize,brand_embedDimension= brand_embedDimension, HCP_vocabsize=HCP_vocabsize,HCP_embedDimension=HCP_embedDimension,disposition_vocabsize=disposition_vocabsize,disposition_embedDimension=disposition_embedDimension,concat_size=concat_size, seq_length = max_seq_length, hidden_dim=hidden_dim, n_layers=1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(1, n_epochs + 1):
	total_loss=0
	for i in range(batches+1):
		optimizer.zero_grad()
		output, hidden = model(Brand_feats[i*batch_size:(i+1)*batch_size],Channel_feats[i*batch_size:(i+1)*batch_size],HHCP_feats[i*batch_size:(i+1)*batch_size],Disposition_feats[i*batch_size:(i+1)*batch_size])
		loss = criterion(output, Y[i*batch_size:(i+1)*batch_size])
		total_loss+=loss.item()
		loss.backward() 
		optimizer.step()
		# print(output)

	print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
	print("Loss: {:.4f}".format(total_loss))

for name, param in model.named_parameters():
    if name =='embedLayer_h.weight':
        print(param.data)

print(len(Embedding_final))
torch.save(Embedding_final,'embedding_weights_RNN.pt')

# a=torch.load('embedding_weights_RNN.pt')
# print(a)

with open('HCP_index_mapping.pkl','wb') as f:
	pickle.dump(hcp_index,f)

# with open('HCP_index_mapping.pkl','rb') as f:
# 	new=pickle.load(f)
# print(new)




