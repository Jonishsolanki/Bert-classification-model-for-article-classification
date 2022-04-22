from multiprocessing.connection import answer_challenge
from optparse import Values
import torch
from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from bert import evaluate
from transformers import BertForSequenceClassification
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import re
from sklearn.metrics import accuracy_score
# from bert import evaluate,accuracy_per_class

import torch as pt
from transformers import BertTokenizer, BertModel




# df = pd.read_csv('valid.csv')
df=pd.read_excel('input.xlsx',sheet_name="Sheet5")
#df=df.fillna('pqrstw empty cell wtsrqp',inplace=True)

df['classes']=['LinkedIn impressions activity']*df.shape[0]
#df.to_csv('cl.csv')
# df.rename(columns={"Target_Name":"classes"},inplace=True)
#df.head()

numtoclass = {10:'Annual Financial reports',
 0:'Awards Recognition Achievement',
 5:'Donation Philanthropy',
 4:'Event Seminar',
 6:'Fund Raise Investment',
 8:'Interview Podcast',
 9:'Joining Promotion Experience',
 2:'LinkedIn impressions activity',
 1:'Merger Acquisition Partnership',
 7:'New Expansion',
 3:'Product Service Launch',
 11:'Not Applicable!!'}

df['classes']= df['classes'].apply(lambda x:re.sub("/"," ",str(x)))
df['WholeText']= df['WholeText'].apply(lambda x:re.sub("/"," ",str(x)))
# possible_labels = df.classes.unique()

possible_labels ={
 'Annual Financial reports',
 'Awards Recognition Achievement',
 'Donation Philanthropy',
 'Event Seminar',
 'Fund Raise Investment',
 'Interview Podcast',
 'Joining Promotion Experience',
 'LinkedIn impressions activity',
 'Merger Acquisition Partnership',
 'New Expansion',
 'Product Service Launch'
}

label_dict = {}
for index, possible_label in enumerate(possible_labels):
    label_dict[possible_label] = index
label_dict

df['label'] = df.classes.replace(label_dict)


# df['data_type'] = ['not_set']*df.shape[0]

# df.groupby(['WholeText', 'classes', 'data_type']).count()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
                                  
 

encoded_data_val = tokenizer.batch_encode_plus(
    df.WholeText.values, 
    add_special_tokens=True, 
    return_attention_mask=True, 
    pad_to_max_length=True, 
    max_length=256, 
    return_tensors='pt'
)

input_ids_val = encoded_data_val['input_ids']
attention_masks_val = encoded_data_val['attention_mask']
labels_val = torch.tensor(df.label.values)

dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)


batch_size = 3

dataloader_validation = DataLoader(dataset_val, 
                                   sampler=SequentialSampler(dataset_val), 
                                   batch_size=batch_size)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=len(label_dict),
                                                      output_attentions=False,
                                                      output_hidden_states=False)

# 

model.load_state_dict(torch.load('data_volume/finetuned_BERT_epoch_7.model', map_location=torch.device('cuda')))

model.to(device)

print('loaded model')

_, predictions, true_vals = evaluate(dataloader_validation,model,device)
print('eval done')
#accuracy_per_class(predictions, true_vals)
#pred=pd.DataFrame(predictions)

#pred.to_csv('pred.csv')

pr=np.argmax(predictions,1)
dr=np.amax(predictions,1)

val=pd.DataFrame(dr,columns=['Val'])

k=[]
for x in predictions:
    if np.amax(x)<4.8:
        k.append(11)
    else:
        for y in range(0,10):
            if x[y]==np.amax(x):
                k.append(y)
                break


adf=pd.DataFrame(k,columns=['classes'])
adf['classes']=adf['classes'].map(numtoclass)

# fadf=pd.DataFrame({'WholeText':df.WholeText,'True_Classes':df.classes,'Classes':adf.classes,'Values':val.Val})
fadf=pd.DataFrame({'WholeText':df.WholeText,'Classes':adf.classes})

# ac=accuracy_score(fadf['True_Classes'],fadf['Classes'])
# print('accuracy is:',ac)
fadf.to_csv('Result/result6.csv')