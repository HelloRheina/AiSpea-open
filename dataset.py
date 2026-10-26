import json
from torch.utils.data import Dataset, DataLoader, random_split
import torch

class DialogueDataset(Dataset):
    """
    Dataset for a JSON file whose root has the structure:
    {
      [
        {
          "metadata": {...},
          "dialog": [
              {"role": "...", "text": "..."},
              ...
          ],
          "labels": {...}
        },
        ...
      ]
    }

    Each sample returned corresponds to ONE utterance (a single
    element of a dialog list).  A flat index is mapped to
    (conversation_idx, utterance_idx) internally.
    """
    def __init__(self, file_path, transform=None):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.conversations = data
        self.transform = transform

        # Build a flat index → (conversation_id, utterance_id) map
        self._idx_map = []
        for c_idx, conv in enumerate(self.conversations):
            for u_idx, _ in enumerate(conv.get("dialog", [])):                
                self._idx_map.append((c_idx, u_idx))

    def __len__(self):
        return len(self._idx_map)

    def __getitem__(self, idx):
        conv_idx, utt_idx = self._idx_map[idx]
        conv = self.conversations[conv_idx]

        sample = {
            "metadata": conv.get("metadata", {}),
            "dialog_item": conv["dialog"][utt_idx],
            "labels": conv.get("labels", {})
        }

        if self.transform:
            sample = self.transform(sample)
        return sample


def load_dialogue_data(file_path, test_ratio=0.2, batch_size=32, shuffle=True, transform=None, collate_fn=None, seed=42):
    """
    Returns train_loader and test_loader with a simple random split.

    Args:
        test_ratio (float): Fraction of samples for the test set.
    """
    ds = DialogueDataset(file_path, transform=transform)
    test_len = int(len(ds) * test_ratio)
    train_len = len(ds) - test_len
    train_ds, test_ds = random_split(
        ds, [train_len, test_len],
        generator=torch.Generator().manual_seed(seed)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, test_loader

# Example usage:
if __name__ == "__main__":
    train_loader, test_loader = load_dialogue_data(
        "data/child_multimodal_dataset.json", test_ratio=0.2, batch_size=2, shuffle=True
    )

    for batch in train_loader:
        print(batch)