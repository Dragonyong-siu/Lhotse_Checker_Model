def get_train_val_Loader(df, train_idx, val_idx, batch_size = 8):
  train_df = df.iloc[train_idx]
  val_df = df.iloc[val_idx]
      
  train_Loader = torch.utils.data.DataLoader(
      LCM_Dataset(train_df),
      batch_size = batch_size,
      shuffle = True,
      num_workers = 2,
      drop_last = True)
    
  val_Loader = torch.utils.data.DataLoader(
      LCM_Dataset(val_df),
      batch_size = batch_size,
      shuffle = False,
      num_workers = 2)
    
  dataloaders_dict = {"train" : train_Loader, "val" : val_Loader}

  return dataloaders_dict
