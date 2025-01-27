
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import torchvision
from torch.utils.data import Subset
import torch
from torch.utils.data import Dataset
from collections import namedtuple
from torchvision import datasets
import os
import argparse
import random
import random
import torch
from torch.utils.data import Dataset


class mul_loader:
    @staticmethod
    def _get_data_loader(dataset, batch_size, num_workers, shuffle=False):
        return data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )

    @staticmethod
    def load_data(dataset):

        if dataset == 'cifar10':
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            train_set = torchvision.datasets.CIFAR10(
                root='./data/cifar10', train=True, download=True, transform=transform_train)
            test_set = torchvision.datasets.CIFAR10(
                root='./data/cifar10', train=False, download=True, transform=transform_test)

            return train_set, test_set

        if dataset == 'cifar100':
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            train_set = torchvision.datasets.CIFAR100(
                root='./data/cifar100', train=True, download=True, transform=transform_train)
            test_set = torchvision.datasets.CIFAR100(
                root='./data/cifar100', train=False, download=True, transform=transform_test)

            return train_set, test_set

        if dataset == 'svhn':
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
            ])

            train_set = torchvision.datasets.SVHN(
                root='./data/svhn', split='train', download=True, transform=transform_train)
            test_set = torchvision.datasets.SVHN(
                root='./data/svhn', split='test', download=True, transform=transform_test)

            return train_set, test_set

        if dataset == 'TinyImageNet':

            train_transform = transforms.Compose([
                transforms.Resize(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

            val_transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

            dataset_dir = './data/tiny-imagenet-200/'
            full_train_dataset = datasets.ImageFolder(os.path.join(dataset_dir, 'train'), transform=train_transform)

            num_train = len(full_train_dataset)
            indices = list(range(num_train))
            split = int(np.floor(0.1 * num_train))  # Using 20% of the data for validation

            np.random.seed(42)  # Set the seed for reproducibility
            np.random.shuffle(indices)
            train_idx, val_idx = indices[split:], indices[:split]
            train_dataset = Subset(full_train_dataset, train_idx)
            val_dataset = Subset(full_train_dataset, val_idx)
            train_dataset.dataset.transform = train_transform  # Apply training transforms
            val_dataset.dataset.transform = val_transform

            return train_dataset, val_dataset

    @staticmethod
    def load_mul_data(dataset, task, f_label=0, forget_size=4500, chunk_size=1000, start_index=0):
        split_data = {}
        if dataset == 'cifar10':

            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            if task == 'class-wise':
                train_set = torchvision.datasets.CIFAR10(
                    root='./data/cifar10', train=True, download=True, transform=transform_train)

                split_data['forget'], split_data['remain'] = LabelSplitter(train_set, label=f_label).split()
                if forget_size is not None:
                    result = RandomSplitter(split_data['forget'], num_samples=forget_size).split()
                    split_data['forget'] = result.selected_data
                    split_data['forget_index'] = result.selected_indices
                    split_data['remain_index'] = result.remaining_indices
                    split_data['remain'] = data.ConcatDataset([result.remaining_data, split_data['remain']])

                #print("forget indexes are", split_data['forget_index'])

            if task == 'selective':

                train_set = torchvision.datasets.CIFAR10(
                    root='./data/cifar10', train=True, download=True, transform=transform_train)
                num_data = int(forget_size)
                result = RandomSplitter(train_set, num_samples=num_data).split()
                split_data['forget'] = result.selected_data
                split_data['remain'] = result.remaining_data
                split_data['forget_index'] = result.selected_indices
                split_data['remain_index'] = result.remaining_indices
                #print("forget indexes are", split_data['forget_index'])

            if task == 'identify':
                # use the vulnerable splitter
                train_set = torchvision.datasets.CIFAR10(
                    root='./data/cifar10', train=True, download=True, transform=transform_train)
                result = VulnerableSplitter(train_set, chunk_size, start_index).split()
                split_data['forget'] = result.selected_data
                split_data['remain'] = result.remaining_data
                split_data['forget_index'] = result.selected_indices
                split_data['remain_index'] = result.remaining_indices

            return split_data


        #######################################
        if dataset == 'cifar100':

            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            if task == 'class-wise':
                train_set = torchvision.datasets.CIFAR100(
                    root='./data/cifar100', train=True, download=True, transform=transform_train)

                split_data['forget'], split_data['remain'] = LabelSplitter(train_set, label=f_label).split()

            if task == 'selective':

                train_set = torchvision.datasets.CIFAR100(
                    root='./data/cifar100', train=True, download=True, transform=transform_train)
                num_data = int(forget_size)
                result = RandomSplitter(train_set, num_samples=num_data).split()
                split_data['forget'] = result.selected_data
                split_data['remain'] = result.remaining_data
                split_data['forget_index'] = result.selected_indices
                split_data['remain_index'] = result.remaining_indices

            return split_data

        #######################################

        if dataset == 'svhn':

            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
            ])

            if task == 'class-wise':
                train_set = torchvision.datasets.SVHN(
                    root='./data/svhn', split='train', download=True, transform=transform_train)

                split_data['forget'], split_data['remain'] = LabelSplitter(train_set, label=f_label).split()

            if task == 'selective':

                train_set = torchvision.datasets.SVHN(
                    root='./data/svhn', split='train', download=True, transform=transform_train)
                print("train size is")
                print(len(train_set))
                num_data = int(forget_size)
                result = RandomSplitter(train_set, num_samples=num_data).split()
                split_data['forget'] = result.selected_data
                split_data['remain'] = result.remaining_data
                split_data['forget_index'] = result.selected_indices
                split_data['remain_index'] = result.remaining_indices

            return split_data

        #############################################
        if dataset == 'TinyImageNet':

            train_transform = transforms.Compose([
                transforms.Resize(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

            val_transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

            if task == 'class-wise':
                dataset_dir = './data/tiny-imagenet-200/'
                train_set = torchvision.datasets.ImageFolder(os.path.join(dataset_dir, 'train'),
                                                             transform=train_transform)
                split_data['forget'], split_data['remain'] = LabelSplitter(train_set, label=f_label).split()

            if task == 'selective':
                dataset_dir = './data/tiny-imagenet-200/'
                train_set = torchvision.datasets.ImageFolder(os.path.join(dataset_dir, 'train'),
                                                             transform=train_transform)
                num_data = int(forget_size)
                result = RandomSplitter(train_set, num_samples=num_data).split()
                split_data['forget'] = result.selected_data
                split_data['remain'] = result.remaining_data
                split_data['forget_index'] = result.selected_indices
                split_data['remain_index'] = result.remaining_indices

            return split_data

    @staticmethod
    def load_selective_data(dataset, indices_file_path, forget_size):
        # Load the dataset using the dataset loader function
        split_data = {}
        train_set, test_set = mul_loader.load_data(dataset)
        if indices_file_path is not None:
            indices_data = torch.load(indices_file_path)
            vulnerable_indices = indices_data['vulnerable']
            print(vulnerable_indices)
            remaining_indices = indices_data['remaining']
            vulnerable_train_set = Subset(train_set, vulnerable_indices)
            remaining_train_set = Subset(train_set, remaining_indices)
            result = RandomSplitter(vulnerable_train_set, num_samples=forget_size).split()
            split_data['forget'] = result.selected_data
            split_data['remain'] = torch.utils.data.ConcatDataset([result.remaining_data, remaining_train_set])
            split_data['forget_index'] = result.selected_indices
            split_data['remain_index'] = result.remaining_indices
            print("forget size is", len(split_data['forget']))
            print("remain size is", len(split_data['remain']))
            print("forget index is", split_data['forget_index'])
            # print labels of the forget data
            subset_loader = torch.utils.data.DataLoader(split_data['forget'],
                                                        batch_size=len(split_data['forget']), shuffle=False)
            #for data, labels in subset_loader:
                #print("forget labels are", labels)
        return split_data

    @staticmethod
    def load_mixed_data(dataset, vulnerable_file_path, privacy_file_path):
        # Load the dataset using the dataset loader function
        split_data = {}
        train_set, test_set = mul_loader.load_data(dataset)
        all_indices = list(range(len(train_set)))

        if vulnerable_file_path is not None:
            vulnerable_indices = torch.load(vulnerable_file_path)
            privacy_indices = torch.load(privacy_file_path)
            privacy_indices = privacy_indices['vulnerable']
            vulnerable_indices = vulnerable_indices['vulnerable']
            #print("vulnerable indices are", len(vulnerable_indices))
            #print("privacy indices are", len(privacy_indices))
            num_vulnerable = min(len(vulnerable_indices), 600)
            num_privacy = min(len(privacy_indices), 600)
            sampled_vulnerable = np.random.choice(vulnerable_indices, num_vulnerable, replace=False).tolist()
            sampled_privacy = np.random.choice(privacy_indices, num_privacy, replace=False).tolist()
            #print(sampled_vulnerable)
            #print(sampled_privacy)
            forget_indices = sampled_vulnerable + sampled_privacy
            remained_indices = list(set(all_indices) - set(forget_indices))
            forget_train_set = Subset(train_set, forget_indices)
            remaining_train_set = Subset(train_set, remained_indices)
            split_data['forget'] = forget_train_set
            split_data['vulnerable'] = Subset(train_set, sampled_vulnerable)
            split_data['privacy'] = Subset(train_set, sampled_privacy)
            split_data['remain'] = remaining_train_set
            split_data['privacy_index'] = sampled_privacy
            split_data['vulnerable_index'] = sampled_vulnerable
            split_data['forget_index'] = forget_indices
            split_data['remain_index'] = remained_indices


        return split_data

    @staticmethod
    def load_mixed_vulnerable_data(dataset, vulnerable_file_path, privacy_file_path):
        split_data = {}
        train_set, test_set = mul_loader.load_data(dataset)
        all_indices = list(range(len(train_set)))
        vulnerable_indices = torch.load(vulnerable_file_path)
        privacy_indices = torch.load(privacy_file_path)
        privacy_indices = privacy_indices['vulnerable']
        vulnerable_indices = vulnerable_indices['vulnerable']
        num_vulnerable = min(len(vulnerable_indices), 600)
        sampled_vulnerable = np.random.choice(vulnerable_indices, num_vulnerable, replace=False).tolist()
        forget_indices = sampled_vulnerable
        random_indices = list(set(all_indices) - set(forget_indices))
        sample_random = np.random.choice(random_indices, 600, replace=False).tolist()
        random_train_set = Subset(train_set, random_indices)
        forget_indices = sampled_vulnerable + sample_random
        forget_train_set = Subset(train_set, forget_indices)
        remaining_indices = list(set(all_indices) - set(forget_indices))
        #remain_train_set = Subset(train_set, remaining_indices)
        split_data['forget'] = forget_train_set
        split_data['random'] = random_train_set
        split_data['remain'] = Subset(train_set, remaining_indices)
        split_data['forget_index'] = forget_indices
        split_data['random_index'] = sample_random
        split_data['remain_index'] = remaining_indices
        return split_data

























    @staticmethod
    def load_test_data(dataset, task='selective', f_label=0):

        split_data_test = {}
        if dataset == 'cifar10':
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            if task == 'class-wise':
                test_set = torchvision.datasets.CIFAR10(
                    root='./data/cifar10', train=False, download=True, transform=transform_test)
                split_data_test['forget'], split_data_test['remain'] = LabelSplitter(test_set, label=f_label).split()
                return split_data_test['remain']

            if task == 'selective' or task == 'vulnerable':
                test_set = torchvision.datasets.CIFAR10(
                    root='./data/cifar10', train=False, download=True, transform=transform_test)
                return test_set

        if dataset == 'cifar100':
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            if task == 'class-wise':
                test_set = torchvision.datasets.CIFAR100(
                    root='./data/cifar100', train=False, download=True, transform=transform_test)
                split_data_test['forget'], split_data_test['remain'] = LabelSplitter(test_set, label=f_label).split()
                return split_data_test['remain']

            if task == 'selective' or task == 'vulnerable':
                test_set = torchvision.datasets.CIFAR100(
                    root='./data/cifar100', train=False, download=True, transform=transform_test)
                return test_set

        if dataset == 'svhn':
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
            ])
            if task == 'class-wise':
                test_set = torchvision.datasets.SVHN(
                    root='./data/svhn', split='test', download=True, transform=transform_test)
                split_data_test['forget'], split_data_test['remain'] = LabelSplitter(test_set, label=f_label).split()
                return split_data_test['remain']

            if task == 'selective' or task == 'vulnerable':
                test_set = torchvision.datasets.SVHN(
                    root='./data/svhn', split='test', download=True, transform=transform_test)
                return test_set

        if dataset == 'TinyImageNet':

            val_transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

            if task == 'class-wise':
                dataset_dir = './data/tiny-imagenet-200/'
                val_dataset = datasets.ImageFolder(os.path.join(dataset_dir, 'val'), transform=val_transform)
                split_data_test['forget'], split_data_test['remain'] = LabelSplitter(val_dataset, label=f_label).split()
                return split_data_test['remain']

            if task == 'selective' or task == 'vulnerable':
                dataset_dir = './data/tiny-imagenet-200/'
                val_dataset = datasets.ImageFolder(os.path.join(dataset_dir, 'val'), transform=val_transform)
                return val_dataset

    @staticmethod
    def poisson_subsample(dataset, lam):
        """ Subsample the dataset based on Poisson distribution. """
        num_samples = len(dataset)
        keep_indices = np.where(np.random.poisson(lam, num_samples) > 0)[0]
        return Subset(dataset, keep_indices)
    @staticmethod
    def uniform_subsample(dataset, size):
        """ Uniformly subsample the dataset. """
        num_samples = len(dataset)
        keep_indices = torch.randperm(num_samples)[:size]
        return Subset(dataset, keep_indices)


class RandomSplitter:
    def __init__(self,
                  data: Dataset, num_samples: int = 0):

        self.data = data
        self.num_samples = num_samples
        self.device = torch.device("cuda:0" if torch.cuda.is_available()
                                    else "cpu")

    def split(self) -> [Dataset, Dataset]:
        SplitResult = namedtuple('SplitResult',
                                 ['selected_data', 'remaining_data', 'selected_indices',
                                  'remaining_indices'])
        if self.num_samples > 0:
            random_indices = torch.randperm(len(self.data))[:self.num_samples]
            remaining_indices = torch.tensor([i for i in range(len(self.data))
                                             if i not in random_indices])
            selected_data = torch.utils.data.Subset(self.data, random_indices)
            remaining_data = torch.utils.data.Subset(self.data,
                                                     remaining_indices)
        else:
            selected_data = None
            remaining_data = self.data
            random_indices = []
            remaining_indices = list(range(len(self.data)))

        return SplitResult(selected_data, remaining_data, random_indices, remaining_indices)


class LabelSplitter:

    def __init__(self, data: Dataset, label: int):
        self.data = data
        self.label = label
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def split(self) -> Dataset:
        if self.label is None:
            return self.data, None
        else:
            mask = torch.tensor([target == self.label for target in self.data.targets])
            selected_indices = torch.where(mask)[0]
            remaining_indices = torch.where(~mask)[0]
            selected_data = torch.utils.data.Subset(
                self.data, selected_indices)
            remaining_data = torch.utils.data.Subset(
                self.data, remaining_indices)

        return selected_data, remaining_data


class VulnerableSplitter:

    def __init__(self, data: Dataset, chunk_size: int = 1000, start_index: int = 0, device=None):
        self.data = data
        self.chunk_size = chunk_size
        self.current_index = start_index  # Counter to track chunk processing
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.all_indices = list(range(len(data)))
        random.shuffle(self.all_indices)  # Shuffle the indices for randomness

    def split(self):
        # Define the SplitResult namedtuple similar to other splitters
        SplitResult = namedtuple('SplitResult',
                                 ['selected_data', 'remaining_data', 'selected_indices', 'remaining_indices'])

        # Check if there are enough samples left for a split
        if self.current_index >= len(self.all_indices):
            return SplitResult(None, self.data, [], list(range(len(self.data))))  # No more data to split

        # Get the next chunk of indices
        end_index = min(self.current_index + self.chunk_size, len(self.all_indices))
        selected_indices = self.all_indices[self.current_index:end_index]

        # Update current index for the next iteration
        self.current_index = end_index

        # Create subsets for the selected data and the remaining data
        selected_data = torch.utils.data.Subset(self.data, selected_indices)
        remaining_indices = self.all_indices[end_index:]
        remaining_data = torch.utils.data.Subset(self.data, remaining_indices)

        # Return the selected chunk and the remaining data in the same format as the other splitters
        return SplitResult(selected_data, remaining_data, selected_indices, remaining_indices)

    def reset(self):
        # Reshuffle the data and reset the counter for another pass
        random.shuffle(self.all_indices)
        self.current_index = 0


class ChunkSplitter:
    def __init__(self, data: Dataset, chunk_size: int = 1000, device=None):
        self.data = data
        self.chunk_size = chunk_size
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.all_indices = list(range(len(data)))  # All available indices
        random.shuffle(self.all_indices)  # Shuffle for randomness
        self.current_index = 0  # Keep track of the current global index

    def split(self):
        # Check if there are enough samples left for a split
        if self.current_index >= len(self.all_indices):
            return {
                'forget': None,
                'remain': None,
                'forget_indices': [],
                'remaining_indices': []
            }

        end_index = min(self.current_index + self.chunk_size, len(self.all_indices))
        forget_indices = self.all_indices[self.current_index:end_index]
        self.current_index = end_index
        remain_indices = [idx for idx in range(len(self.data)) if idx not in forget_indices]
        forget_data = torch.utils.data.Subset(self.data, forget_indices)
        remain_data = torch.utils.data.Subset(self.data, remain_indices)
        result = {
            'forget': forget_data,
            'remain': remain_data,
            'forget_indices': forget_indices,
            'remaining_indices': remain_indices
        }
        return result