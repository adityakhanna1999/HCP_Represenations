import torch
from torch import nn
import numpy as np

class Model(nn.Module):
	def __init__(self, channel_vocabsize, channel_embedDimension, brand_vocabsize, brand_embedDimension, HCP_vocabsize, HCP_embedDimension, disposition_vocabsize,disposition_embedDimension,concat_size, hidden_dim, seq_length, n_layers=1, output_size=5):
		super(Model, self).__init__()

		self.hidden_dim = hidden_dim
		self.n_layers = n_layers

		self.embedLayer_c = nn.Embedding(channel_vocabsize, channel_embedDimension)
		self.embedLayer_b = nn.Embedding(brand_vocabsize, brand_embedDimension)
		self.embedLayer_h = nn.Embedding(HCP_vocabsize, HCP_embedDimension)
		self.embedLayer_d = nn.Embedding(disposition_vocabsize, disposition_embedDimension)

		self.rnn = nn.RNN(concat_size, hidden_dim, n_layers, batch_first=True)   
		
		self.fc = nn.Linear(hidden_dim*seq_length, output_size)
		self.sof = nn.Softmax(dim=1)


	
	def forward(self,brand_feats,channel_feats,HCP_feats,disposition_feats):

		# All feats are of the format [batch_size * sequence_length]{contains indexes of brands}
		# All embed are in the format of [batch_size * sequence_length * Embedding_size]
		# Concat embedding is now [Batch_size * sequence_length * Total Embedding Size]

		batch_size=brand_feats.size()[0]
		brand_embed=self.embedLayer_b(brand_feats)
		channel_embed=self.embedLayer_c(channel_feats)
		HCP_embed=self.embedLayer_h(HCP_feats)
		disposition_embed=self.embedLayer_d(disposition_feats)
		print(brand_embed.size(),"1")

		concat_embedding=torch.cat((brand_embed,channel_embed,HCP_embed,disposition_embed),dim=2)

		hidden = self.init_hidden(batch_size)
		out, hidden = self.rnn(concat_embedding, hidden)
		out = out.contiguous().view(batch_size, -1)
		out = self.fc(out)
		# out=self.sof(out)
		return out, hidden
	
	def init_hidden(self, batch_size):

		hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
		return hidden