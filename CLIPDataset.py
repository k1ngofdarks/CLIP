import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image
import typing as tp


class CLIPDataset(Dataset):
    def __init__(
        self,
        image_path: str,
        image_filenames: tp.Sequence[str],
        captions: tp.Sequence[str],
        tokenizer,
        max_tokenizer_length: int = 200,
    ):
        """Dataset for CLIP training.

        Args:
            image_path: Directory with image files.
            image_filenames: Image file names relative to ``image_path``.
            captions: Text captions aligned with ``image_filenames``.
            tokenizer: HuggingFace tokenizer.
            max_tokenizer_length: Maximum caption length in tokens.
        """
        if len(image_filenames) != len(captions):
            raise ValueError(
                "image_filenames and captions must have the same length. "
                f"Got {len(image_filenames)} and {len(captions)}"
            )

        self.image_path = image_path
        self.image_filenames = list(image_filenames)
        self.captions = list(captions)

        self.encoded_captions = tokenizer(
            self.captions,
            max_length=max_tokenizer_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        self.transforms = T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __getitem__(self, idx: int) -> tp.Dict[str, tp.Union[torch.Tensor, str]]:
        image_name = self.image_filenames[idx]
        img = Image.open(f"{self.image_path}/{image_name}").convert("RGB")
        img = self.transforms(img)

        item = {key: values[idx] for key, values in self.encoded_captions.items()}
        item["image"] = img
        item["caption"] = self.captions[idx]
        return item

    def __len__(self):
        return len(self.captions)
