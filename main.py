import os
import random
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import lr_scheduler
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torch.nn import Softmax
from transformers import BigBirdConfig, BigBirdModel, BigBirdTokenizer, AdamW

os.environ["WANDB_DISABLED"] = "true"
torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

intents = {
	"greetings": ["Hi!", "Hello", "How are you doing?"],
	
	"thank_you": ["You are welcome!", "Happy to help!", "That's what I'm here for!"],
	
	"mail": ["Please restart the Outlook",
	         "Please try to check your OWA",
	         "Please see below instruction for troubleshooting the mailbox"],
	
	"screen": ["For screen related issues I will need you to please send me the Serial Number.",
	           "Then kindly follow the below instruction: ",
	           "Please see if all cables are connected"],
	
	"keyboard": ["Please try first cleaning your keyboard.",
	             "Please try to unplug the keyboard and plug it back in",
	             "It seems like I will need to call a technician to get it fixed locally."],
	
	"software": ["Please press 'control', 'alt' and 'delete', find the application and end it.",
	             "Please now restart the application",
	             "Please see if you are using the most recent version of the application."]
}

# MODEL
checkpoint = 'google/bigbird-roberta-base'

config = BigBirdConfig(max_position_embeddings=16, attention_type='block_sparse')
bigbird = BigBirdModel(config).from_pretrained(checkpoint)


class Model(nn.Module):
	def __init__(self, bigbird_model):
		super(Model, self).__init__()
		self.bigbird_model = bigbird_model
		
		self.dropout = nn.Dropout(0.2)
		self.relu = nn.ReLU()

		self.input_function1 = nn.Linear(768, 512)
		self.input_function2 = nn.Linear(512, 256)
		self.output_function = nn.Linear(256, 6)

		self.softmax = nn.LogSoftmax(dim=1)
	
	def forward(self, sent_id, mask):
		cls_hs = self.bigbird_model(sent_id, attention_mask=mask)[0][:, 0]
		
		x = self.input_function1(cls_hs)
		x = self.relu(x)
		x = self.dropout(x)
		
		x = self.input_function2(x)
		x = self.relu(x)
		x = self.dropout(x)
		x = self.output_function(x)
		
		x = self.softmax(x)

		return x


for param in bigbird.parameters():
	param.requires_grad = False

model = Model(bigbird).to(device)

dataset = pd.read_csv('intents.csv')
le = LabelEncoder()
dataset['categories'] = le.fit_transform(dataset['categories'])


train_examples, train_categories = dataset['examples'], dataset['categories']

tokenizer = BigBirdTokenizer.from_pretrained(checkpoint)
sequence_length = 16


def tokenize(inp):
	result = tokenizer(
		inp.tolist(),
		padding='max_length',
		truncation=True,
		max_length=sequence_length,
		return_token_type_ids=False,
		return_tensors="pt")
	return result


tokens_train = tokenize(train_examples)

# Zamień na tensory

train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
print(train_seq, train_mask)

train_y = torch.tensor(train_categories.tolist())
train_data = TensorDataset(train_seq, train_mask, train_y)

batch_size = 16

train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

class_wts = compute_class_weight('balanced', classes=np.unique(train_categories), y=train_categories)
weights = torch.tensor(class_wts, dtype=torch.float)
weights = weights.to(device)

# Loss function
cross_entropy = nn.NLLLoss(weight=weights)

# Optimizer i learning rate scheduler
learning_rate = 0.01
optimizer = AdamW(model.parameters(), lr=learning_rate)
lr_sch = lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)


# TRENING
def train():
	model.train()
	total_loss = 0
	total_preds = []

	for step, batch in enumerate(train_dataloader):
		
		batch = [r.to(device) for r in batch]
		sent_id, mask, labels = batch
		
		preds = model(sent_id, mask)
		
		loss = cross_entropy(preds, labels)
		total_loss = total_loss + loss.item()
		
		loss.backward()
		
		# clip the gradients to 1.0. It helps in preventing the exploding gradient problem
		# update parameters
		
		torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
		
		optimizer.step()
		optimizer.zero_grad()
		
		lr_sch.step()
		
		# model predictions are stored on GPU. So, push it to CPU
		preds = preds.detach().cpu().numpy()
		
		# append the model predictions
		total_preds.append(preds)
	
	# compute the training loss of the epoch
	avg_loss = total_loss / len(train_dataloader)
	
	# predictions are in the form of (no. of batches, size of batch, no. of classes).
	# reshape the predictions in form of (number of samples, no. of classes)
	total_preds = np.concatenate(total_preds, axis=0)
	# returns the loss and predictions
	return avg_loss, total_preds


train_losses = []
epochs = 400

for epoch in range(epochs):
	train_loss, _ = train()
	train_losses.append(train_loss)
	print(f'Epoch {epoch + 1}/{epochs}')
	print(f'Loss function value: {train_loss:.3f}')


torch.save(model, 'trained_model.pt')


torch.save(model.state_dict(), 'trained_model_state.pt')

plt.plot(range(epochs), train_losses)
plt.show()


# ROZPOZNAWANIE KATEGORRI

def find_intent(message):
	message = re.sub(r'[^a-zA-Z ]+', '', message)
	message = [message]

	tokenized_message = tokenizer(message)

	test_seq = torch.tensor(tokenized_message['input_ids'])
	test_mask = torch.tensor(tokenized_message['attention_mask'])

	with torch.no_grad():
		preds = model(test_seq.to(device), test_mask.to(device))
	
	# Soft max - prawdopodobieństwo kategorii
	soft_it = nn.Softmax(dim=1)
	soft_preds = soft_it(preds)
	soft_preds = soft_preds.detach().cpu().numpy()
	soft_preds = max(soft_preds[0])

	preds = preds.detach().cpu().numpy()
	preds = np.argmax(preds, axis=1)

	return le.inverse_transform(preds)[0], soft_preds


# DIALOG
print('Hi, welcome to your new virtual assistant. I am here to help you with your IT related issues. :)')

while True:
	question = input("Please type your question.\n")
	
	counter = 0
	while True:
		answer, confidence = find_intent(question)
		counter += 1
		if confidence >= 0.7 or counter == 10:
			break
	
	print('Kategoria: ', f'{answer}, pewność: {confidence}')
	
	if answer is None or answer == '':
		print("I'm sorry, I don't understand your question. Please try to retype it.\n")
	elif confidence <= 0.7:
		print("I am not sure what do you mean, please try again. \n")
	else:
		print('Odpowiedź: ', random.choice(intents[answer]))
	
	if answer != 'greetings' and confidence >= 0.7:
		ask_if_end = input("Is there anything more I can help you with? [Yes|No]\n")
		if ask_if_end.lower() == 'yes':
			continue
		elif ask_if_end.lower() == 'no':
			print("Have a good day!")
			break
