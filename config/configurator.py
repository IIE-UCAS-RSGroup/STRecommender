# -*- coding: utf-8 -*-
import re
import os
import sys
import yaml

class Config(object):
    def __init__(self, config_file_list = None):
        self.yaml_loader = self._build_yaml_loader()
        self.file_config_dict = self._load_config_files(config_file_list)
        self.cmd_config_dict = self._load_cmd_line()
        self.final_config_dict = self._get_final_config_dict()

    def _build_yaml_loader(self):
        loader = yaml.FullLoader
        loader.add_implicit_resolver(
            u'tag:yaml.org,2002:float',
            re.compile(
                u'''^(?:
             [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$''', re.X
            ), list(u'-+0123456789.')
        )
        return loader

    def _load_config_files(self, file_list):
        file_config_dict = dict()
        if file_list:
            for file in file_list:
                with open(file, 'r', encoding='utf-8') as f:
                    file_config_dict.update(yaml.load(f.read(), Loader=self.yaml_loader))
        return file_config_dict

    def _convert_config_dict(self, config_dict):
        r"""This function convert the str parameters to their original type.
        """
        for key in config_dict:
            param = config_dict[key]
            if not isinstance(param, str):
                continue
            try:
                value = eval(param)
                if value is not None and not isinstance(value, (str, int, float, list, tuple, dict, bool, Enum)):
                    value = param
            except (NameError, SyntaxError, TypeError):
                if isinstance(param, str):
                    if param.lower() == "true":
                        value = True
                    elif param.lower() == "false":
                        value = False
                    else:
                        value = param
                else:
                    value = param
            config_dict[key] = value
        return config_dict

    def _load_cmd_line(self):
        r""" Read parameters from command line and convert it to str.
        """
        cmd_config_dict = dict()
        if "ipykernel_launcher" not in sys.argv[0]:
            for i in range(1, len(sys.argv)):
                if sys.argv[i].startswith("--"):
                    cmd_arg_name, cmd_arg_value = sys.argv[i][2:], sys.argv[i + 1]
                    i += 1
                if cmd_arg_name in cmd_config_dict and cmd_arg_value != cmd_config_dict[cmd_arg_name]:
                    raise SyntaxError("There are duplicate commend arg '%s' with different value." % arg)
                else:
                    cmd_config_dict[cmd_arg_name] = cmd_arg_value
        cmd_config_dict = self._convert_config_dict(cmd_config_dict)
        return cmd_config_dict

    def _get_final_config_dict(self):
        final_config_dict = dict()
        final_config_dict.update(self.file_config_dict)
        final_config_dict.update(self.cmd_config_dict)

        model_yaml_file = './properties/model/' + final_config_dict['model'] + '.yaml'
        temp = []
        temp.append(model_yaml_file)
        model_para = self._load_config_files(temp)
        for key, value in model_para.items():
            if key not in final_config_dict.keys():
                final_config_dict[key] = value

        return final_config_dict

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        self.final_config_dict[key] = value

    def __getattr__(self, item):
        if 'final_config_dict' not in self.__dict__:
            raise AttributeError(f"'Config' object has no attribute 'final_config_dict'")
        if item in self.final_config_dict:
            return self.final_config_dict[item]
        raise AttributeError(f"'Config' object has no attribute '{item}'")

    def __getitem__(self, item):
        if item in self.final_config_dict:
            return self.final_config_dict[item]
        else:
            return None