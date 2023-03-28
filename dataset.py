import functools
import itertools
import random
import torch
from utils import *


class CIRDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            image_base_path,
            data_split_paths
    ):
        self.image_base_path = image_base_path

        self.cir_samples = list(
            itertools.chain.from_iterable(
                [
                    load_file(path)
                    for path in data_split_paths
                ]
            )
        )

    def __len__(self):
        return len(self.cir_samples)

    def __getitem__(
            self,
            index
    ):
        cir_sample = self.cir_samples[index]
        reference = random.choice(cir_sample["references"])

        reference_image = get_image(
            os.path.join(
                self.image_base_path,
                reference["image"]
            )
        )

        reference_text = reference["text"]
        modification_text = cir_sample["modification_text"]
        target = random.choice(cir_sample["targets"])

        target_image = get_image(
            os.path.join(
                self.image_base_path,
                target["image"]
            )
        )

        target_text = target["text"]
        return reference_image, reference_text, modification_text, target_image, target_text

    def get_cir_loader(
            self,
            batch_size,
            image_extractor,
            text_tokenizer
    ):
        cir_loader = torch.utils.data.DataLoader(
            self,
            batch_size,
            True,
            num_workers=2,
            collate_fn=functools.partial(
                CIRBatch.collate,
                image_extractor=image_extractor,
                text_tokenizer=text_tokenizer
            ),
            pin_memory=True
        )

        return cir_loader

    def get_retrieval_data(
            self,
            sample_count
    ):
        indices = range(len(self))

        if sample_count > 0:
            indices = random.sample(
                indices,
                sample_count
            )

        image_lookup = {}
        unique_images = []
        group_lookup = {}
        image_groups = []
        queries = []
        labels = []

        for index in indices:
            reference_image, reference_text, modification_text, target_image, target_text = self[index]

            if reference_image.path not in image_lookup:
                image_lookup[reference_image.path] = len(image_lookup)
                unique_images.append(reference_image)

                if reference_text not in group_lookup:
                    group_lookup[reference_text] = len(group_lookup)

                image_groups.append(group_lookup[reference_text])

            queries.append(
                (
                    image_lookup[reference_image.path],
                    modification_text
                )
            )

            if target_image.path not in image_lookup:
                image_lookup[target_image.path] = len(image_lookup)
                unique_images.append(target_image)

                if target_text not in group_lookup:
                    group_lookup[target_text] = len(group_lookup)

                image_groups.append(group_lookup[target_text])

            labels.append(image_groups[image_lookup[target_image.path]])

        return queries, labels, unique_images, image_groups


class CIRBatch:
    def __init__(
            self,
            reference_image_features,
            reference_text_tokens,
            reference_text_mask,
            modification_text_tokens,
            modification_text_mask,
            target_image_features,
            target_text_tokens,
            target_text_mask
    ):
        self.reference_image_features = reference_image_features
        self.reference_text_tokens = reference_text_tokens
        self.reference_text_mask = reference_text_mask
        self.modification_text_tokens = modification_text_tokens
        self.modification_text_mask = modification_text_mask
        self.target_image_features = target_image_features
        self.target_text_tokens = target_text_tokens
        self.target_text_mask = target_text_mask

    @classmethod
    def collate(
            cls,
            cir_samples,
            image_extractor,
            text_tokenizer
    ):
        reference_images, reference_texts, modification_texts, target_images, target_texts = zip(*cir_samples)

        reference_image_extraction = image_extractor(
            list(reference_images),
            return_tensors="pt"
        )

        reference_text_tokenization = text_tokenizer(
            list(reference_texts),
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        modification_text_tokenization = text_tokenizer(
            list(modification_texts),
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        target_image_extraction = image_extractor(
            list(target_images),
            return_tensors="pt"
        )

        target_text_tokenization = text_tokenizer(
            list(target_texts),
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        cir_batch = cls(
            reference_image_extraction["pixel_values"],
            reference_text_tokenization["input_ids"],
            reference_text_tokenization["attention_mask"],
            modification_text_tokenization["input_ids"],
            modification_text_tokenization["attention_mask"],
            target_image_extraction["pixel_values"],
            target_text_tokenization["input_ids"],
            target_text_tokenization["attention_mask"]
        )

        return cir_batch

    def pin_memory(self):
        self.reference_image_features = self.reference_image_features.pin_memory()
        self.reference_text_tokens = self.reference_text_tokens.pin_memory()
        self.reference_text_mask = self.reference_text_mask.pin_memory()
        self.modification_text_tokens = self.modification_text_tokens.pin_memory()
        self.modification_text_mask = self.modification_text_mask.pin_memory()
        self.target_image_features = self.target_image_features.pin_memory()
        self.target_text_tokens = self.target_text_tokens.pin_memory()
        self.target_text_mask = self.target_text_mask.pin_memory()
        return self

    def to(self, *args, **kwargs):
        self.reference_image_features = self.reference_image_features.to(*args, **kwargs)
        self.reference_text_tokens = self.reference_text_tokens.to(*args, **kwargs)
        self.reference_text_mask = self.reference_text_mask.to(*args, **kwargs)
        self.modification_text_tokens = self.modification_text_tokens.to(*args, **kwargs)
        self.modification_text_mask = self.modification_text_mask.to(*args, **kwargs)
        self.target_image_features = self.target_image_features.to(*args, **kwargs)
        self.target_text_tokens = self.target_text_tokens.to(*args, **kwargs)
        self.target_text_mask = self.target_text_mask.to(*args, **kwargs)
        return self
