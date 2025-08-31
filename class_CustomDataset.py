


class CustomDataset(Dataset):
    def __init__(self, train_text, tokenizer):
        """
        Initialize the dataset with data and targets.
        Args:
            data: The input data (e.g., features).
            targets: The corresponding labels or targets.
        """
        self.train_text = train_text 
        self.tokenizer = tokenizer
    def __len__(self):
        """
        return the total number of samples.
        """
        return len(self.train_text)

    def __getitem__(self, idx):
        """
        Retrieve a sample and its target at the given index
        """
        return self.train_text[idx], self.tokenizer[idx]

    # Example usage
    data = torch.tensor(([1,2], [3,4], [5,6]))
    targets = torch.tensor([0,1,0])
    dataset = CustomDataset(data, targets)

    print("Number of samples: ", len(dataset))
    print("First sample: ", dataset[0])
    