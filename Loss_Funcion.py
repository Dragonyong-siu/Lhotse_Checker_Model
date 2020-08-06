def Loss_Function(Start_logits, End_logits, Start_idx, End_idx):
  loss_function = nn.CrossEntropyLoss()
  Front_loss = loss_function(Start_logits, Start_idx)
  Behind_loss = loss_function(End_logits, End_idx)
  Total_loss = Front_loss + Behind_loss
  return Total_loss
