import csv
import os
import json
import re

from abc import ABC, abstractmethod
from utils import PAD_char, SOS_char, EOS_char, PAD_token, SOS_token, EOS_token



class BaseVocabulary(ABC):

    def __init__(self,
                 name: str 
                 ):
        self.name = name
        self.char2index = {}
        self.index2char = {}
        self.n_chars = 0
        self.char2count = {}

    @abstractmethod
    def addSequence(self, sequence):
        pass

    def addchar(self, char):
        if char not in self.char2index:
            self.char2index[char] = self.n_chars
            self.index2char[self.n_chars] = char
            self.char2count[char] = 1
            self.n_chars += 1
        else:
            self.char2count[char] +=1

    def to_dict(self): 
        return {
                'name' : self.name,
                'char2index' : self.char2index,
                'index2char' : {str(key) : value for key,value in self.index2char.items()}, 
                }

    @classmethod
    @abstractmethod
    def from_dict(cls, data):
        pass



class SmilesVocabulary(BaseVocabulary):
    def __init__(self,
                 name
                 ):
        super().__init__(name)
        self.char2index = {PAD_char: PAD_token, SOS_char: SOS_token, EOS_char: EOS_token}
        self.index2char = {PAD_token: PAD_char, SOS_token: SOS_char, EOS_token: EOS_char}
        self.n_chars = 3
        self.regex_pattern = ( # https://projects.volkamerlab.org/teachopencadd/talktorials/T034_recurrent_neural_networks.html
            r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\." 
            r"|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"
        )
        self.regex = re.compile(self.regex_pattern)

    def addSequence(self, smiles):
        for char in self.regex.findall(smiles):
            self.addchar(char)

    @classmethod
    def from_dict(cls,data):
        vocab = cls(data['name'])
        vocab.char2index = data['char2index']
        vocab.index2char = {int(key) : value for key,value in data['index2char'].items()}
        vocab.n_chars = data['n_chars']
        vocab.char2count = {char: 0 for char in vocab.char2index if char not in [PAD_char, SOS_char, EOS_char]}
        return vocab

def save_vocabulary(vocab, filename):
    """Saves the vocabulary to a JSON file."""
    try:
        dir_name = os.path.dirname(filename)
        if dir_name: 
             os.makedirs(dir_name, exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(vocab.to_dict(), f, indent=4)
        print(f"Vocabulary saved to {filename}")
    except Exception as e:
        print(f"Error saving vocabulary to {filename}: {e}")

def load_vocabulary(filename):
    """Loads the vocabulary from a JSON file."""
    if not os.path.exists(filename):
        return None
    try:
        with open(filename, 'r') as f:
            vocab_data = json.load(f)
        vocab = SmilesVocabulary.from_dict(vocab_data)
        return vocab
    except Exception as e:
        print(f"Error loading vocabulary from {filename}: {e}")
        return None

