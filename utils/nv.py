import dataclasses
import json
import logging
import os
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf, UnsupportedValueType
from ruamel import yaml as ruamel_yaml
from simple_parsing import ArgumentParser, ConflictResolution

from utils.non_nv import run_bash

@dataclass
class CommandLineOpts:
    """
    Base class to parse commandline args using a dataclass and simple_parsing
    """

    defaults_yaml = 'none'
    """ A YAML file that contains the commandline defaults.
        'none' indicates no YAML file is used
        Note: To allow taking an arg default from the YAML file, its value should be set to YAML_DEFAULT
    """

    @classmethod
    def get_opts(cls, ignore_unknown_args=False):
        """ Usage example:
            opts = CommandLineOpts.get_opts()
        """
        parser = ArgumentParser(conflict_resolution=ConflictResolution.NONE)
        parser.add_arguments(cls, dest='cfg')
        if ignore_unknown_args:
            parsed_args: cls = parser.parse_known_args()[0].cfg
        else:
            parsed_args: cls = parser.parse_args().cfg

        if parsed_args.defaults_yaml == 'none':
            opts = parsed_args
        else:
            # load defaults from YAML
            opts = OmegaConf.load(parsed_args.defaults_yaml)
            # override defaults by commandline args
            opts = cls.override_defaults(dataclasses.asdict(parsed_args), opts)
            opts = cls.call_post_process_opts(opts, parsed_args)

        return opts

    @staticmethod
    def post_process_opts(opts):
        # Override this method with your post-processing code
        return opts

    @classmethod
    def call_post_process_opts(cls, opts, parsed_args):
        # override a dict with defaults by commandline args
        for key, value in vars(parsed_args).items():
            if dataclasses.is_dataclass(value) and 'post_process_opts' in dir(value):
                opts[key] = value.__getattribute__('call_post_process_opts')(opts[key], parsed_args.__getattribute__(key))
        opts = cls.post_process_opts(opts)
        return opts

    @classmethod
    def override_defaults(cls, parsed_args: dict, defaults):
        # override a dict with defaults by commandline args
        for key, value in parsed_args.items():
            if isinstance(parsed_args[key], dict):
                if key in defaults:
                    defaults[key] = cls.override_defaults(parsed_args[key], defaults[key])
                else:
                    try:
                        defaults[key] = parsed_args[key]
                    except UnsupportedValueType:
                        msg = f'Missing default value for field name "{key}" in YAML file'
                        raise RuntimeError(msg)

            else:
                if value != YAML_DEFAULT:
                    defaults[key] = value
                else:
                    pass

        return defaults


class YAML_DEFAULT():
    """ This is used to indicate an arg default is set by the YAML defaults file"""
    pass


def comma_seperated_str_to_list(comma_seperated_str, regex_sep=r', |[, ]'):
    return re.split(regex_sep, comma_seperated_str)


def slice_dict_to_dict(d, keys, returned_keys_prefix='', returned_keys_postfix='', ignore_missing_keys=False):
    """ Returns a tuple from dictionary values, ordered and slice by given keys
        keys can be a list, or a CSV string
    """
    if isinstance(keys, str):
        keys = keys[:-1] if keys[-1] == ',' else keys
        keys = re.split(', |[, ]', keys)

    if returned_keys_prefix != '' or returned_keys_postfix != '':
        return OrderedDict((returned_keys_prefix + k + returned_keys_postfix, d[k]) for k in keys)

    if ignore_missing_keys:
        return OrderedDict((k, d[k]) for k in keys if k in d)
    else:
        return OrderedDict((k, d[k]) for k in keys)

def sorted_tuple(*args):
    return tuple(sorted(args))



def fill_missing_by_defaults(args_dict, argparse_parser):
    all_arg_keys = get_all_argparse_keys(argparse_parser)
    for key in all_arg_keys:
        if key not in args_dict:
            args_dict[key] = argparse_parser.get_default(key)
    return args_dict


def get_all_argparse_keys(parser):
    return [action.dest for action in parser._actions]

def to_json(args_dict, log_dir, filename):
    args_json = os.path.join(log_dir, filename)
    with open(args_json, 'w') as f:
        json.dump(args_dict, f)
        print(f'\nDump configuration to JSON file: {args_json}\n\n')


def load_ordered_yaml(yaml_fname):
    # using ruamel to load a YAML while keeping the order and visual structure of the file
    with open(yaml_fname) as f:
        experiment_yaml = ruamel_yaml.load(f, Loader=ruamel_yaml.RoundTripLoader)
    return experiment_yaml

def get_uncommitted_files(list_ignored_regex_pattern_filenames, ignore_untracked_files):
    ignore_untracked_files_str = ''
    if ignore_untracked_files:
        ignore_untracked_files_str = ' -uno'
    if list_ignored_regex_pattern_filenames is None:
        list_ignored_regex_pattern_filenames = []
    git_status = run_bash('git status --porcelain' + ignore_untracked_files_str)
    uncommitted_files = []
    for line in git_status.split('\n'):
        ignore_current_file = False
        if line:
            fname = re.split(' ?\w +', line)[1]
            for ig_file_regex_pattern in list_ignored_regex_pattern_filenames:
                if re.match(ig_file_regex_pattern, fname):
                    ignore_current_file = True
                    break
            if ignore_current_file:
                continue
            else:
                uncommitted_files.append(fname)
    return uncommitted_files

def append_to_text_file(txt, fname, end='\n'):
    with Path(fname).open('a') as f:
        f.write(txt + end)

def is_uncommited_git_repo(list_ignored_regex_pattern_filenames=None, ignore_untracked_files=True):
    """ Check if there are uncommited files in workdir.
        Can ignore specific file names or regex patterns
    """

    uncommitted_files_list = get_uncommitted_files(list_ignored_regex_pattern_filenames, ignore_untracked_files)

    if uncommitted_files_list:
        return True
    else:
        return False


def count_down(T, cont_on_any_key=False, msg=''):
    """
    :type T: int
    """
    if msg:
        print(msg)
    else:
        print('\nWaiting for %d seconds. press ctrl-C to break' % T, end='')
        if cont_on_any_key:
            print(', or any key to continue')

    try:
        print('')
        for t in range(T):
            if T-t<=5:
                print(f'{T-t}', end='', flush=True)
            _, exit_code = run_bash('read -n 1 -t 1', return_exist_code=True)
            if cont_on_any_key and exit_code == 0:
                return
            # time.sleep(1)
        print('')
    except KeyboardInterrupt:
        exit(1)


