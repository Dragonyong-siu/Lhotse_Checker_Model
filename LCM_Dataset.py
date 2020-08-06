class LCM_Dataset(torch.utils.data.Dataset):
  def __init__(self, df, max_len = 110):
    self.df = df
    self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', 
                                                   do_lower_case=True)
    self.max_len = max_len
    self.labeled = 'selected_text' in df
  
  def __len__(self):
    return len(self.df)
  
  def __getitem__(self, idx):
    Data = {}
    self.idx = idx
    row = self.df.iloc[idx]
    input_ids, token_type_ids, attention_mask, sentence = self.get_input_data(row)
    Data['input_ids'] = input_ids
    Data['token_type_ids'] = token_type_ids
    Data['sentence'] = sentence
    Data['attention_mask'] = attention_mask
    
    if self.labeled:
      Start_idx, End_idx = self.get_target_idx(row)
      Data['Start_idx'] = Start_idx
      Data['End_idx'] = End_idx
      
    return Data

  def get_input_data(self, row):
    text_ex = row['text']
    text_ex = str(text_ex)
    sentence= " " + " ".join(text_ex.split())
    
    ##encoding
    input_ids = self.tokenizer.encode(text_ex)
    token_type_ids = self.tokenizer(text_ex).token_type_ids
    attention_mask = self.tokenizer(text_ex).attention_mask
    
    ##Padding
    padding_length = self.max_len - len(input_ids)
    input_ids = input_ids + [1] * padding_length
    token_type_ids = token_type_ids + [0] * padding_length
    attention_mask = attention_mask + [0] * padding_length

    ##Tensoring
    input_ids = torch.Tensor(input_ids).long()
    token_type_ids = torch.Tensor(token_type_ids).long()
    attention_mask = torch.Tensor(attention_mask).long()

    return input_ids, token_type_ids, attention_mask, sentence

  def get_target_idx(self, row):
    text_ex = row['text']
    selected_ex = row['selected_text']
    k = self.idx
    text_token = self.tokenizer.tokenize(text_ex)
    selected_token = self.tokenizer.tokenize(selected_ex)
    len_token = len(text_token)
    
    TOF = [any([id in selected_token for id in ids.split()]) for ids in text_token]
  
    if True not in TOF:
      TOF = [True] * len_token
    
    entire_len = len(TOF)
    start_idx = TOF.index(True)
    RE_TOF = list(reversed(TOF))
    end_idx = RE_TOF.index(True)
    end_idx = entire_len - end_idx
     
    Start_idx = start_idx
    End_idx = end_idx - 1     
    
    return Start_idx, End_idx
