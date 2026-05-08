import copy
import logging
import pickle

import math
import os
import random
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset, DataLoader, default_collate
from transformers import AutoTokenizer, VisionEncoderDecoderModel, ViTImageProcessor, AutoModel, \
    ChineseCLIPImageProcessor, CLIPProcessor
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageFile
from tqdm import tqdm
from scipy.fftpack import fft, dct

ImageFile.LOAD_TRUNCATED_IMAGES = True

chinese_datasets = ["weibo_CN", "weibo21"]


def load_specified_dataset(model_name, config):
    return FHMEXDataset


class BaseMultiModalityFakeNewsDataset(Dataset):
    chinese_datasets = ['weibo_CN', 'weibo21']

    def __init__(self, config, data, additional_data_dict=None, mode='train'):
        super(BaseMultiModalityFakeNewsDataset, self).__init__()
        assert mode in ['train', 'eval', 'test'], 'Mode must be train, eval or test !'
        self.mode = mode
        self.config = config
        self.data_path = config.data_path
        self.img_root = os.path.join(config.data_path, 'images')
        self.dataset = config.dataset
        self.max_text_len = config.max_text_len
        key_column = ['image', 'text', 'label', 'event']

        for column in key_column:
            assert column in data.columns, f"Column '{column}' not found in {mode} data !"

        self.data = data

        self.additional_data = additional_data_dict
        self.batch_dict = {}

    def clean_data(self, data):
        data['text'] = data['text'].fillna(' ')
        data['image'] = data['image'].str.lower()

        return data

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image, text, label = row['image'], row['text'], row['label']

        return (image, text, label)

    def __len__(self):
        return len(self.data)

    def collate_fn(self, x):
        image, text, label = default_collate(x)
        self.batch_dict['image'] = image
        self.batch_dict['text'] = text
        self.batch_dict['label'] = label
        return self.batch_dict

    def update(self):
        pass

    @staticmethod
    def load_additional_data(config, data_path):
        return {}

    @staticmethod
    def image_name_mapping(data_path):
        img_name_mapping_path = f'{data_path}/img_name_mapping.pkl'

        if os.path.exists(img_name_mapping_path):
            img_name_mapping = pickle.load(open(img_name_mapping_path, 'rb'))
            logging.info(f'Loaded {len(img_name_mapping)} image tensors from {img_name_mapping_path}.')
        else:
            img_name_mapping = {}
            for img_name in tqdm(os.listdir(os.path.join(data_path, 'images')), desc='Loading images'):
                img_name, img_type = img_name.split('/')[-1].split(".")
                img_name_lower = img_name.lower() + '.' + img_type
                img_name_mapping[img_name_lower] = img_name

            pickle.dump(img_name_mapping, open(img_name_mapping_path, 'wb'))
            logging.info(f'Saved {len(img_name_mapping)} image name mapping to {img_name_mapping_path}.')

        return img_name_mapping


def encode_batch(batch_text, tokenizer, max_length, padding='longest'):
    try:
        outputs = tokenizer(
            batch_text,
            max_length=max_length,
            padding=padding,
            return_tensors='pt',
            truncation=True,
        )
        input_ids = outputs["input_ids"]
        attention_mask = outputs["attention_mask"]
    except:
        input_ids = torch.zeros((len(batch_text), max_length), dtype=torch.long)
        attention_mask = torch.zeros((len(batch_text), max_length), dtype=torch.long)

    return input_ids, attention_mask


class MMFakeNewsDataset(BaseMultiModalityFakeNewsDataset):
    def __init__(self, config, data, additional_data_dict=None, mode='train'):
        super(MMFakeNewsDataset, self).__init__(config, data, additional_data_dict, mode)
        self.model_name = config.model
        self.image_size = config.image_size
        self.data_path = config.data_path
        self.tokenizer = additional_data_dict['tokenizer']
        self.img_name2tensor = additional_data_dict['img_name2tensor']

    @staticmethod
    def language_is_chinese(dataset):
        is_chinese = dataset in chinese_datasets
        return is_chinese

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image = self.select_image(row['image'])
        image, text, label = self.img_name2tensor[image], row['text'], row['label']

        cur_tensors = (image,
                       text,
                       torch.tensor(label, dtype=torch.long))

        return cur_tensors

    def select_image(self, images: str):

        imgs = images.split("|")
        random.shuffle(imgs)

        for img in imgs:
            if img in self.img_name2tensor:
                return img
        raise ValueError(f'No valid image found for {images}')

    def collate_fn(self, x):
        image, text, label = default_collate(x)
        text_input_ids, text_attention_mask = encode_batch(list(text), self.tokenizer, self.max_text_len)

        self.batch_dict['image'] = image
        self.batch_dict['text_input_ids'] = text_input_ids
        self.batch_dict['text_attention_mask'] = text_attention_mask
        self.batch_dict['label'] = label
        return self.batch_dict

    @staticmethod
    def load_image(data_path):
        img_tensor_dict_path = f'{data_path}/img_name2tensor.pkl'

        if os.path.exists(img_tensor_dict_path):
            img_name2tensor = pickle.load(open(img_tensor_dict_path, 'rb'))
            logging.info(f'Loaded {len(img_name2tensor)} image tensors from {img_tensor_dict_path}.')
        else:
            data_transforms = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            img_name2tensor = {}
            image_dir = os.path.join(data_path, 'images')
            for img_name in tqdm(os.listdir(image_dir), desc='Loading images'):
                if img_name == '802451c9tw1dtqmr5prdlj.jpg':
                    print(img_name)
                # try:
                im = Image.open(os.path.join(image_dir, img_name)).convert('RGB')
                im = data_transforms(im)
                try:
                    img_name_prefix, img_type = img_name.split('/')[-1].split(".")
                    img_name_lower = img_name_prefix.lower() + '.' + img_type
                    img_name2tensor[img_name_lower] = im
                except Exception as e:
                    print(e)
                    print(f'invalid image: {img_name}')

            pickle.dump(img_name2tensor, open(img_tensor_dict_path, 'wb'))
            logging.info(f'Saved {len(img_name2tensor)} image tensors to {img_tensor_dict_path}.')

        return img_name2tensor

    @staticmethod
    def load_additional_data(config, data_path):
        # return {'tokenizer': None}
        additional_data_dict = {}

        lan_type = 'chinese' if MMFakeNewsDataset.language_is_chinese(config.dataset) else 'uncased'
        additional_data_dict['tokenizer'] = AutoTokenizer.from_pretrained(
            f'/mnt1/userhome/proj/LLMs/bert-base-{lan_type}')
        additional_data_dict['img_name2tensor'] = MMFakeNewsDataset.load_image(data_path)

        return additional_data_dict


class FHMEXDataset(MMFakeNewsDataset):
    def __init__(self, config, data, additional_data_dict=None, mode='train'):
        super(FHMEXDataset, self).__init__(config, data, additional_data_dict, mode)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image = self.select_image(row['image'])
        image, text, label = self.img_name2tensor[image], row['text'], row['label']

        cur_tensors = (image,
                       text,
                       torch.tensor(label, dtype=torch.long))

        return cur_tensors

    def collate_fn(self, x):
        if FHMEXDataset.language_is_chinese(self.dataset):
            return self.collate_fn_chinese(x)
        return self.collate_fn_english(x)

    def collate_fn_chinese(self, data):
        image, text, label = default_collate(data)

        token_chinese = self.tokenizer
        token_data = token_chinese.batch_encode_plus(
            batch_text_or_text_pairs=text,
            truncation=True,
            padding="max_length",
            max_length=self.max_text_len,
            return_tensors="pt",
            return_length=True,
        )

        self.batch_dict['image'] = image
        self.batch_dict['input_ids'] = token_data["input_ids"]
        self.batch_dict['attention_mask'] = token_data["attention_mask"]
        self.batch_dict['token_type_ids'] = token_data["token_type_ids"]
        self.batch_dict['clip_inputs'] = None
        self.batch_dict['label'] = label

        return self.batch_dict

    def collate_fn_english(self, data):

        image, text, label = default_collate(data)

        token_english = self.tokenizer
        token_data = token_english.batch_encode_plus(
            batch_text_or_text_pairs=text,
            truncation=True,
            padding="max_length",
            max_length=self.max_text_len,
            return_tensors="pt",
            return_length=True,
        )

        self.batch_dict['image'] = image
        self.batch_dict['input_ids'] = token_data["input_ids"]
        self.batch_dict['attention_mask'] = token_data["attention_mask"]
        self.batch_dict['token_type_ids'] = token_data["token_type_ids"]
        self.batch_dict['clip_inputs'] = None
        self.batch_dict['label'] = label

        return self.batch_dict

    @staticmethod
    def load_image(data_path):
        img_tensor_dict_path = f'{data_path}/img_name2tensor_only_resize.pkl'

        if os.path.exists(img_tensor_dict_path):
            img_name2tensor = pickle.load(open(img_tensor_dict_path, 'rb'))
            logging.info(f'Loaded {len(img_name2tensor)} image tensors from {img_tensor_dict_path}.')
        else:
            data_transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])

            img_name2tensor = {}
            image_dir = os.path.join(data_path, 'images')
            for img_name in tqdm(os.listdir(image_dir), desc='Loading images'):
                im = Image.open(os.path.join(image_dir, img_name)).convert('RGB')
                im = data_transforms(im)
                try:
                    img_name_prefix, img_type = img_name.split('/')[-1].split(".")
                    img_name_lower = img_name_prefix.lower() + '.' + img_type
                    img_name2tensor[img_name_lower] = im
                except Exception as e:
                    print(e)
                    print(f'invalid image: {img_name}')

            pickle.dump(img_name2tensor, open(img_tensor_dict_path, 'wb'))
            logging.info(f'Saved {len(img_name2tensor)} image tensors to {img_tensor_dict_path}.')

        return img_name2tensor

    @staticmethod
    def load_additional_data(config, data_path):
        additional_data_dict = {}
        lan_type = 'chinese' if FHMEXDataset.language_is_chinese(config.dataset) else 'uncased'
        additional_data_dict['tokenizer'] = AutoTokenizer.from_pretrained(
            f'/mnt1/userhome/proj/LLMs/bert-base-{lan_type}')
        additional_data_dict['img_name2tensor'] = FHMEXDataset.load_image(data_path)

        return additional_data_dict
