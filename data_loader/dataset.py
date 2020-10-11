import modin.pandas as pd
from torchtext.data import TabularDataset


class ChatbotDataset(TabularDataset):
    """Chatbot dataset."""

    def __init__(self, csv_file, data_dir, path, format, fields, transform=None, **kwargs):
        super().__init__(path, format, fields, **kwargs)
        self.chatbot_frame = pd.read_csv(csv_file, delimiter='\t')
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return self.chatbot_frame.size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample
