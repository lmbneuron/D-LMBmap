from torch.utils.data.dataset import Dataset
import json
import os
from random import randint, shuffle
from copy import deepcopy
import torch
from ..core.tensor_dict import tensor_cpu

from ..constrain import ReaderZoo


class Brain(Dataset):
    def __init__(self,
                 data_json: str,
                 constrain: dict):
        """
        :param data_json: json path that contains data info
        :param constrain: such as
                        {
                              simi: True
                              outline: True
                              convex: False
                              hole_pointcloud: True
                        }
        """
        super(Brain, self).__init__()
        with open(data_json) as f:
            content = json.load(f)
        moving_list = content["moving"]
        fix_list = content["fix"]
        self.constrain = constrain
        self.reader = {}
        for k, v in self.constrain.items():
            if v:
                self.reader[k] = ReaderZoo.get_reader_by_constrain(k)
        self.moving_prefix_list = [os.path.join(i, os.path.split(i)[1])
                                  for i in moving_list]
        self.fix_prefix_list = [os.path.join(i, os.path.split(i)[1])
                               for i in fix_list]
        print("len :", len(self.moving_prefix_list))

    def __getitem__(self, index):
        """
        :param index:
        :return: dict: {
                    fix: {
                        name:  brain name of fixed image
                        simi (processed fixed image）：{
                            img: image data
                        }
                        outline (mask of outline): {
                            img: data of outline
                        }
                        hole (mask of ventricle) : {
                            img: data of ventricle
                        }
                        ...
                    }
                    mov: {
                        name:  brain name of moving image
                        simi (processed moving image）：{
                            img: image data
                        }
                        outline (marked outline of moving image): {
                            img: data of outline
                        }
                        hole (marked ventricle of moving image) : {
                            img: data of ventricle
                        }
                        ...
                    }
                }
        """

        mov_prefix = self.moving_prefix_list[index]
        fix_prefix = self.fix_prefix_list[index]
        output = {"fix": {"name": os.path.split(fix_prefix)[1]},
                  "mov": {"name": os.path.split(mov_prefix)[1]}}
        for k, v in self.reader.items():
            output["fix"][k] = self.reader[k](fix_prefix)
            output["mov"][k] = self.reader[k](mov_prefix)
        return output

    def __len__(self):
        return len(self.moving_prefix_list)


class BaseTemplate(Dataset):
    """
    dataset when training average template
    """
    def __init__(self,
                 data_json: str,
                 constrain: dict):
        """
        :param data_json: json path that contains data info
        :param constrain: such as
                        {
                              simi: True
                              outline: True
                              convex: False
                              hole_pointcloud: True
                        }
        """
        super(BaseTemplate, self).__init__()
        with open(data_json) as f:
            content = json.load(f)
        data_list = content["fix"]
        self.constrain = constrain
        self.init_template_prefix = os.path.join(content["moving"][0], os.path.split(content["moving"][0])[1])
        self.reader = {}
        for k, v in self.constrain.items():
            if v:
                self.reader[k] = ReaderZoo.get_reader_by_constrain(k)
        self.data_prefix_list = [os.path.join(i, os.path.split(i)[1])
                                  for i in data_list]
        print("len :", len(self.data_prefix_list))

    def __len__(self):
        return len(self.data_prefix_list)

    def __getitem__(self, index):
        pass


class TwoTemplate(BaseTemplate):
    """
    return fix image and moving image
    """
    def __init__(self, data_json: str, constrain: dict):
        super(TwoTemplate, self).__init__(data_json, constrain)

    def __getitem__(self, index):
        """
        :param index:
        :return: dict: {
                    fix: {
                        name:  brain name of fixed image
                        simi (processed fixed image）：{
                            img: image data
                        }
                        outline (mask of outline): {
                            img: data of outline
                        }
                        hole (mask of ventricle) : {
                            img: data of ventricle
                        }
                        ...
                    }
                    mov: {
                        name:  brain name of moving image
                        simi (processed moving image）：{
                            img: image data
                        }
                        outline (marked data of outline): {
                            img: data of outline
                        }
                        hole (marked data of ventricle) : {
                            img: data of ventricle
                        }
                        ...
                    }
                }
        """

        mov_prefix = self.data_prefix_list[index]
        fix_prefix = self.data_prefix_list[randint(0, len(self.data_prefix_list)-1)]
        output = {"fix": {"name": os.path.split(fix_prefix)[1]},
                  "mov": {"name": os.path.split(mov_prefix)[1]}}
        for k, v in self.reader.items():
            output["fix"][k] = self.reader[k](fix_prefix)
            output["mov"][k] = self.reader[k](mov_prefix)
        return output


class SimpleBrain(Dataset):
    def __init__(self,
                 data_list,
                 constrain: dict):
        """
        This attributes of this class and the Brain is different.
        :param data_json: json path that contains data info
        :param constrain: such as
                        {
                              simi: True
                              outline: True
                              convex: False
                              hole_pointcloud: True
                        }
        """
        super(SimpleBrain, self).__init__()
        self.constrain = constrain
        self.reader = {}
        for k, v in self.constrain.items():
            if v:
                self.reader[k] = ReaderZoo.get_reader_by_constrain(k)
        self.data_prefix_list = [os.path.split(i)[0] for i in data_list]
        self.data_prefix_list = [os.path.join(i, os.path.split(i)[1]) for i in self.data_prefix_list]

    def __getitem__(self, index):
        """
        :param index:
        :return: dict: {
                    fix: {
                        name:  brain name of fixed image
                        simi (processed fixed image）：{
                            img: image data
                        }
                        outline (mask of outline): {
                            img: data of outline
                        }
                        hole (mask of ventricle) : {
                            img: data of ventricle
                        }
                        ...
                    }
                    mov: {
                        name:  brain name of moving image
                        simi (processed moving image）：{
                            img: image data
                        }
                        outline (marked outline of moving image): {
                            img: data of outline
                        }
                        hole (marked ventricle of moving image) : {
                            img: data of ventricle
                        }
                        ...
                    }
                }
        """

        data_prefix = self.data_prefix_list[index]
        output = {"name": os.path.split(data_prefix)[1]}
        for k, v in self.reader.items():
            output[k] = self.reader[k](data_prefix)
        return output

    def __len__(self):
        return len(self.data_prefix_list)
