class Lhotse_Checker_Model(nn.Module):
  def __init__(self):
    super(Lhotse_Checker_Model, self).__init__()
    config= BertConfig.from_pretrained('/content/gdrive/My Drive/bert_config.json', 
                                       output_hidden_states = True)
    self.BertModel = BertModel.from_pretrained("bert-base-uncased", config = config)
    self.dropout = nn.Dropout(0.5)
    self.linear = nn.Linear(config.hidden_size, 2)
    nn.init.normal_(self.linear.weight, std=0.02)
    nn.init.normal_(self.linear.bias, 0)

  def forward(self, input_ids, token_type_ids, attention_mask):
    _, _, hidden_layer = self.BertModel(input_ids, token_type_ids, attention_mask)
    
    X = torch.stack([hidden_layer[-1], hidden_layer[-2], 
                     hidden_layer[-3], hidden_layer[-4]])
    X = torch.mean(X, 0)
    X = self.dropout(X)
    X = self.linear(X)
    
    Start_logits, End_logits = X.split(1, dim=-1)
    Start_logits = Start_logits.squeeze(-1)
    End_logits = End_logits.squeeze(-1)
                
    return Start_logits, End_logits
