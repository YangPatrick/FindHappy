import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm

pretrained_path = '../PretrainedModels/bert-wwm-chinese/'
train_file = 'train.txt'
tokenizer = BertTokenizer.from_pretrained(pretrained_path)

class ClassificationDataset(Dataset):
    def __init__(self,tokenizer,file):
        self.tokenizer = tokenizer
        self.x = []
        self.y = []
        f = open(file)
        lines = f.readlines()
        f.close()
        for line in lines:
            text,label = line.strip().split('\t')
            inputs = self.tokenizer(text,padding='max_length',truncation=True,max_length=510,return_tensors='pt')
            self.x.append({'input_ids':inputs['input_ids'].squeeze(),'attention_mask': inputs['attention_mask'].squeeze()})
            self.y.append(int(label)-1)
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    def __len__(self):
        return len(self.y)

class Classifier(nn.Module):
    def __init__(self, pretrained_path):
        super(Classifier, self).__init__()
        self.model = BertModel.from_pretrained(pretrained_path)
        self.linear = nn.Linear(in_features=768, out_features=5)
        self.cirt = nn.CrossEntropyLoss()
        self.device = None
        #self.parameters = self.linear.parameters

    def forward(self, inputs):
        if self.device is None:
            self.device = next(self.parameters()).device
        output = self.model(input_ids=inputs['input_ids'].cuda(self.device),
                            attention_mask=inputs['attention_mask'].cuda(self.device))
        preds = self.linear(output.pooler_output)
        return preds

    def inference(self, inputs):
        pred = self.forward(inputs)
        return pred.argmax().item()

    def loss(self, inputs, targets):
        preds = self.forward(inputs)
        loss = self.cirt(preds, targets.cuda(self.device))
        return loss

# dataset = ClassificationDataset(tokenizer, train_file)
# torch.save(dataset,'train.pt')
train_dataset = torch.load('train.pt')
dataloader = DataLoader(train_dataset,batch_size=16,shuffle=True)
valid_dataset = torch.load('valid.pt')
# classifier = Classifier(pretrained_path)
classifier = torch.load('classifier.pt')
classifier.cuda(3)
optimizer = optim.Adadelta(classifier.parameters(), lr=0.001)
epochs = 10
global_steps = 0
best_score = 10
with torch.no_grad():
    classifier.eval()
    score = 0
    bar = tqdm(valid_dataset)
    for data in bar:
        inputs,label = data
        inputs = {'input_ids':inputs['input_ids'].unsqueeze(0),'attention_mask': inputs['attention_mask'].unsqueeze(0)}
        pred = classifier.inference(inputs)
        bar.set_description('[!] Evaluating ')
        score += (pred-label)**2
    score /= len(valid_dataset)
    if score < best_score:
        torch.save(classifier,'classifier.pt')
        best_score = score
for epoch in range(epochs):
    classifier.train()
    bar = tqdm(dataloader)
    for index, data in enumerate(bar):
        global_steps += 1
        optimizer.zero_grad()
        inputs, labels = data
        loss = classifier.loss(inputs, labels)
        loss.backward()
        optimizer.step()
        bar.set_description(
            f'[!]Train epoch-{epoch} Step-{global_steps} Score={round(best_score,4)} loss={round(loss.item(),4)}'
        )
    with torch.no_grad():
        classifier.eval()
        score = 0
        bar = tqdm(valid_dataset)
        for data in bar:
            inputs,label = data
            inputs = {'input_ids':inputs['input_ids'].unsqueeze(0),'attention_mask': inputs['attention_mask'].unsqueeze(0)}
            pred = classifier.inference(inputs)
            bar.set_description('[!] Evaluating ')
            score += (pred-label)**2
        score /= len(valid_dataset)
        if score < best_score:
            torch.save(classifier,'classifier.pt')
            best_score = score
