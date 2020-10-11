import modin.pandas as pd
from torchtext.data import TabularDataset, Field


class ChatbotDataset(TabularDataset):
    """Chatbot dataset."""

    def __init__(self, path, format, fields, transform=None, **kwargs):

        self.chatbot_frame = pd.read_csv(csv_file, delimiter='\t')
        self.data_dir = data_dir
        self.transform = transform
        super().__init__(path, format, fields, **kwargs)
