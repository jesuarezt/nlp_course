import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F



class RNN_LSTM_BI(nn.Module):
    def __init__(self, vocab_size, embed_dim, rnn_hidden_size, fc_hidden_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 
                                      embed_dim, 
                                      padding_idx=0) 
        self.rnn = nn.LSTM(embed_dim, rnn_hidden_size, 
                           batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(rnn_hidden_size*2, fc_hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text, lengths):
        out = self.embedding(text)
        out = nn.utils.rnn.pack_padded_sequence(out, lengths.cpu().numpy(), enforce_sorted=False, batch_first=True)
        _, (hidden, cell) = self.rnn(out)
        out = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out
    



import numpy as np
def train_LSTM_BI(dataloader, model, optimizer, loss_func):
    model.train()
    total_acc, total_loss = 0, 0
    losses,acc_acum,nums=0,0,0

    for text_batch, label_batch, lengths in dataloader:
        optimizer.zero_grad()
        y_test=label_batch.long()
        y_pred = torch.argmax(model(text_batch, lengths), 1)

        pred = model(text_batch, lengths)
        loss = loss_func(pred, y_test)
        loss.backward()
        optimizer.step()
  
        total_acc += (y_pred == label_batch).float().sum().item()
        total_loss += loss.item()*label_batch.size(0)

    return total_acc/len(dataloader.dataset), total_loss/len(dataloader.dataset)

def evaluate_LSTM_BI(dataloader, model, loss_func):
    model.eval()
    total_acc, total_loss = 0, 0
    with torch.no_grad():
        for text_batch, label_batch, lengths in dataloader:
            #revisar funcion de perdida
            y_pred = torch.argmax(model(text_batch, lengths), 1)
            y_test=label_batch.long()
            pred = model(text_batch, lengths)
            loss = loss_func(pred, y_test)
            total_acc += (y_pred == label_batch).float().sum().item()
            total_loss += loss.item()*label_batch.size(0)
    return total_acc/len(dataloader.dataset), total_loss/len(dataloader.dataset)


#loss_fn = nn.CrossEntropyLoss()

#loss_func = F.cross_entropy
#optimizer = torch.optim.Adam(model3_1.parameters(), lr=lr)


def train_model_lstm(  vocab_size, embed_dim, rnn_hidden_size, fc_hidden_size, device, num_classes,
                      loss_func, optimizer, lr, num_epochs, train_dl, valid_dl):
    model = RNN_LSTM_BI(vocab_size= vocab_size,
     embed_dim= embed_dim,
      rnn_hidden_size= rnn_hidden_size, fc_hidden_size= fc_hidden_size, num_classes=num_classes) 
    model = model.to(device)
    

    train_losses = []
    valid_losses = []

    for epoch in range(num_epochs):
        if epoch % 10 == 0:
            acc_train, loss_train = train_LSTM_BI(dataloader= train_dl, model = model, optimizer=optimizer, loss_func= loss_func)
            
            acc_valid, loss_valid = evaluate_LSTM_BI(dataloader=valid_dl, model=model, loss_func=loss_func)
            train_losses.append(loss_train)
            valid_losses.append(loss_valid)
            #print(f"epoch: {epoch},    train_loss: {loss_train:.4f} \
            #valid_loss: {loss_valid:.4f}, valid_acc: {acc_valid:.4f}")

    #plt.plot(range(len(train_losses)), train_losses, 'r', label='train')
    #plt.plot(range(len(train_losses)), valid_losses, 'b', label = 'valid')
    #plt.legend()
