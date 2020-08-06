def gain_selected_text(text, Start_idx, End_idx):
  tokens = tokenizer.tokenize(text)
  length_tokens = len(tokens)
  tokens = tokens + (110 - length_tokens)*['a']
  selected_TEXT = []
  for i in range(End_idx - Start_idx + 1):
    selected_word = tokens[i + Start_idx]
    selected_TEXT.append(selected_word)
  
  selected_TEXT = " ".join(selected_TEXT)
  if " ##" in selected_TEXT:
    selected_TEXT = selected_TEXT.replace(" ##", "")

  return selected_TEXT
  
# def jaccard , def Jaccard_score is from sazuma's idea in kaggle(sentiment extraction) not mine. 
def jaccard(str1, str2):
  sent1 = set(str1.lower().split()) 
  sent2 = set(str2.lower().split())
  intersection = sent1.intersection(sent2) 
  
  return float(len(intersection)) / (len(sent1) + len(sent2) - len(intersection))

def Jaccard_score(text, Start_idx, End_idx, Start_logits, End_logits):
  Start_pred = np.argmax(Start_logits)
  End_pred = np.argmax(End_logits)
  if Start_pred > End_pred:
    pred = text
  else:
    pred = gain_selected_text(text, Start_pred, End_pred)
  true = gain_selected_text(text, Start_idx, End_idx)
    
  return jaccard(true, pred)
