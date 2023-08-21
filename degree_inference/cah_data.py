from .text_dataset import TextDataset
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import wordnet
from transformers import BertTokenizer


class CAHData:
    def __init__(self, augment=True, include_gpt_inferences=True, include_ilr=True):
        self.augment = augment
        self.include_gpt_inferences = include_gpt_inferences
        self.include_ilr = include_ilr
        self.df = self.load()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.encoder = LabelEncoder()
        self.encoder.fit(self.df.label.tolist())

    def rescinded_codes(self):
        changes = pd.read_csv('data/CAH_Change_Log.csv')
        rescinded = changes[changes['CAH_Code_Currently_Rescinded?'].str.contains('Yes')]
        rescinded_codes = rescinded['Current_CAH_Code']
        return [s.strip() for s in rescinded_codes]

    def cah3_mapping(self):
        csv = pd.read_csv('data/HECoS_CAH_Mappings.csv')
        return csv.set_index('CAH3_Code')['CAH3_Label'].to_dict()

    def class_weights(self):
        return compute_class_weight('balanced', classes=self.df['label'].unique(), y=list(self.df['label']))

    def ilr_mappings(self):
        # Large manually mapped list from ILR
        df = pd.read_csv('data/ilr_cah.csv')
        df = df.rename(columns={'LDCS_name':"text","CAH3 code":"label"})
        df = df[['text', 'label']]

        # remove names with numbers in as these are likely to be junk
        df = df.dropna(subset=['text'])
        df = df.dropna(subset=['label'])
        df = df[~df['text'].str.contains('\d')]
        df = df[~df['text'].str.contains('\w\w\.')]

        # format CAH codes consistently
        df['label'] = df['label'].str.replace('CAH', '')

        # Remove rescinded codes
        df = df[~df['label'].isin(self.rescinded_codes())]

        return df

    def load(self):
        # Every mapping of degree to CAH3 from HECoS
        csv = pd.read_csv('data/HECoS_CAH_Mappings.csv')
        df = csv.rename(columns={'HECoS_Label':"text","CAH3_Code":"label"})
        df = df[['text', 'label']]

        if(self.include_ilr):
            df = pd.concat([df, self.ilr_mappings()])

        if(self.include_gpt_inferences):
            gpt_inferences = pd.read_csv('data/gpt-4-4k.csv')
            df = pd.concat([df, gpt_inferences])

        df['text'] = df['text'].str.lower()
        df['label'] = df['label'].str.strip()
        return df.sample(frac=1)

    def datasets(self):
        train, test = train_test_split(self.df, test_size=0.2)
    
        def replace_synonym(sentence, prob=0.5):
            words = sentence.split()
            augmented_words = []
        
            for word in words:
                if random.random() < prob:
                    synonyms = wordnet.synsets(word)
                    if synonyms and synonyms[0].lemmas():
                        synonym = random.choice(synonyms[0].lemmas()).name()
                        augmented_words.append(synonym)
                    else:
                        augmented_words.append(word)
                else:
                    augmented_words.append(word)
        
            return ' '.join(augmented_words)
        
        if(self.augment):
            train_augmented = train.copy()
            train_augmented['text'] = train_augmented['text'].apply(replace_synonym)
            train = pd.concat([train, train_augmented], axis=0, ignore_index=True)

        train_texts = train['text'].tolist()
        test_texts = test['text'].tolist()

        train_encodings = self.tokenizer(train_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        test_encodings = self.tokenizer(test_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")

        train_labels = train['label'].tolist()
        test_labels = test['label'].tolist()

        train_labels = self.encoder.transform(train_labels)
        test_labels = self.encoder.transform(test_labels)

        return {
            'train': TextDataset(train_encodings, train_labels),
            'test': TextDataset(test_encodings, test_labels)
        }
