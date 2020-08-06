def train_model(model, dataloaders_dict, criterion, optimizer, num_epochs, filename):
  model.cuda()

  for epoch in range(num_epochs):
    for phase in ['train', 'val']:
      if phase == 'train':
        model.train()
      else:
        model.eval()

      epoch_loss = 0.0
      epoch_jaccard= 0.0

      for data in (dataloaders_dict[phase]):
        input_ids = data['input_ids'].cuda()
        token_type_ids = data['token_type_ids'].cuda()
        attention_mask = data['attention_mask'].cuda()
        sentence = data['sentence']
        Start_idx = data['Start_idx'].cuda()
        End_idx = data['End_idx'].cuda()

        optimizer.zero_grad()

        with torch.set_grad_enabled(phase == 'train'):

          Start_logits, End_logits = model(input_ids, token_type_ids, attention_mask)

          Loss = criterion(Start_logits, End_logits, Start_idx, End_idx)

          if phase == 'train':
            Loss.backward()
            optimizer.step()

          epoch_loss += Loss.item() * len(input_ids)
          Start_idx = Start_idx.cpu().detach().numpy()
          End_idx= End_idx.cpu().detach().numpy()
          Start_logits = torch.softmax(Start_logits, dim=1).cpu().detach().numpy()
          End_logits = torch.softmax(End_logits, dim=1).cpu().detach().numpy()

          for i in range(len(input_ids)):
          jaccard_score= Jaccard_score(
                sentence[i],
                Start_idx[i],
                End_idx[i],
                Start_logits[i], 
                End_logits[i])
            epoch_jaccard += jaccard_score
  
      epoch_loss= epoch_loss / len(dataloaders_dict[phase].dataset)
      epoch_jaccard = epoch_jaccard / len(dataloaders_dict[phase].dataset)

      print('Epoch {}/{} | {:^5} | Loss: {:.4f} | Jaccard: {:.4f}'.format(
          epoch + 1, num_epochs, phase, epoch_loss, epoch_jaccard))
  
  torch.save(model.state_dict(), filename)
