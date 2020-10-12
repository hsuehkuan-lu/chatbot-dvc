import spacy
import modin.pandas as pd
from torchtext.data import TabularDataset, Field


class ChatbotField(Field):
    """Chatbot Field."""
    def __init__(self, lang, **kwargs):
        if 'is_target' in kwargs.keys():

        self.spacy_lang = spacy.load(lang)
        super(ChatbotField, self).__init__(**kwargs)

    def _tokenizer(self, text):
        return [tok.text for tok in self.spacy_lang.tokenizer(text)]


class ChatbotDataset(TabularDataset):
    """Chatbot dataset."""

    def __init__(self, path, format, fields, transform=None, **kwargs):

        self.chatbot_frame = pd.read_csv(csv_file, delimiter='\t')
        self.data_dir = data_dir
        self.transform = transform
        super().__init__(path, format, fields, **kwargs)
