import argparse
import itertools
import json
import os
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import uuid

import yaml
from colorama import Style, Fore
from simple_parsing import field
from torch.utils.hipify.hipify_python import bcolors

from utils.non_nv import run_bash
from utils.nv import load_ordered_yaml, is_uncommited_git_repo, append_to_text_file, count_down, YAML_DEFAULT, \
    CommandLineOpts, get_uncommitted_files
import ruamel.yaml as ruamel_yaml


class CloudTarget(Enum):
    NGC = 'NGC'
    docker = 'docker'


@dataclass
class NGCOpts(CommandLineOpts):
    """ """
    prefix_agent_commandline_ngc: str = field(default=YAML_DEFAULT, help=argparse.SUPPRESS)
    job_name: str = field(default=YAML_DEFAULT, help=argparse.SUPPRESS)
    run_cmd_ngc: str = field(default=YAML_DEFAULT, help=argparse.SUPPRESS)
    num_machines: int = 1  # number of NGC machines to ask
    num_gpus: int = YAML_DEFAULT # number of GPUs in a single NGC machine
    ngc_instance: str = YAML_DEFAULT
    workspace: str = YAML_DEFAULT # NGC workspace name (or hash)
    upload_timeout: int = YAML_DEFAULT

    @staticmethod
    def post_process_opts(opts):
        assert (1 <= opts.num_gpus <= 8)
        opts.ngc_instance = opts.ngc_instance.format(num_gpus=opts.num_gpus)
        return opts


@dataclass
class GitOpts(CommandLineOpts):
    """ """
    commit: str = 'last'
    """ Which commit to clone and sweep on. Default is last. """

    clone_url: str  = field(default=YAML_DEFAULT, help=argparse.SUPPRESS)
    commit_archive_dir: str  = field(default=YAML_DEFAULT, help=argparse.SUPPRESS)

    """ URL to clone the repository (using git clone)"""

    @staticmethod
    def post_process_opts(opts):
        # Command for cloning the top of the repository
        # `|| exit 1` will abort the current machine if the git clone has failed
        if opts.clone_url:
            opts.cmd_clone: str = "git clone {clone_url} --depth={commit_depth} {source_dir} || exit 1"

            # Command for checking out the selected commit.
            # NOTE: If checkout fails, we changedir to /tmp in order to break the consequent execution
            opts.cmd_checkout: str = "cd {source_dir}; git remote set-branches origin {branch}; git fetch -v; " \
                                     "git checkout {commit} || cd /tmp"
        else:
            archive_dir = opts.commit_archive_dir
            assert archive_dir is not None
            opts.cmd_clone: str = "echo" # do nothing
            opts.cmd_checkout: str = "mkdir -p {source_dir}; cd {source_dir}; unzip -o /workspace/%s/{commit}.zip || cd /tmp"%archive_dir

        opts.cmd_commit_depth = "git rev-list HEAD ^{commit} --count"
        if opts.commit == 'last':
            opts.commit = run_bash('git rev-parse --short HEAD')
        all_branches = run_bash(f'git branch -a --contains {opts.commit}').split('\n')
        print(all_branches)
        for branch_ptr in all_branches:
            print(branch_ptr)
            if branch_ptr.startswith('*'):
                opts.branch = branch_ptr.split(' ')[1]
                break
        else:
            print(all_branches)
            raise RuntimeError()

        assert opts.branch is not None

        return opts

@dataclass
class LocalDockerOpts(CommandLineOpts):
    """ """

    prefix_agent_commandline_docker: str = field(default=YAML_DEFAULT, help=argparse.SUPPRESS)
    """ prefix for commandline, when calling the agent on the docker image """

    run_cmd_docker: str = field(default=YAML_DEFAULT, help=argparse.SUPPRESS)
    """ command line template for running the sweep inside a docker image """


# Setting the commandline arguments as a dataclass
@dataclass
class SweepOpts(CommandLineOpts):
    """     """

    git: GitOpts
    """ git options """

    docker: LocalDockerOpts
    """ Local docker options """

    ngc: NGCOpts
    """ NGC options """

    sweep_yaml: str = field(alias="-s")
    """ YAML filename to sweep on """

    defaults_yaml: str = field(alias="-d")
    """ launch configuration yaml e.g. configurations/launch_sweep/train.yaml """

    only_committed: bool = field(default=True)
    """ Only launch when the current repository is committed (i.e. no uncommitted files)"""

    target: CloudTarget = field(default='docker')
    """ target machine: a local docker or NGC instance """

    image_source_code_dir: str  = field(default=YAML_DEFAULT, help=argparse.SUPPRESS)
    """ directory to clone the code into"""

    wandb_key: str = YAML_DEFAULT
    """ 
    W&B login key. Options: default|none|LOGINKEY .
    default: taking the login API key from .netrc at user home dir.
    none: don't use w&b login.
    APIKEY: specifying the login API key.
    """

    experiment_commandline: str = field(default=YAML_DEFAULT, help=argparse.SUPPRESS)
    """ commmand line template for calling the sweep experiment inside a docker image """

    sweep_history_dir: str = field(default=YAML_DEFAULT, help=argparse.SUPPRESS)
    """ Where to save the history of launched sweeps """

    ignore_uncommited_dirs: str = field(default=YAML_DEFAULT, help=argparse.SUPPRESS)
    """ comma separated directory names that are ignored for uncommited check """
    @staticmethod
    def post_process_opts(opts):
        ### Unexposed class attributes

        opts.jupyter_cmd = """./docker_cfg/ts jupyter notebook --ip=0.0.0.0 --allow-root --no-browser  --NotebookApp.token='' --NotebookApp.allow_origin='*' --notebook-dir=/workspace"""
        opts.webhost_cmd = """./docker_cfg/ts bash -c "mkdir -p /workspace/www; cd /workspace/www; /opt/conda/bin/python -m http.server 8000" """
        if opts.wandb_key == 'none':
            opts.wandb_login = 'echo'  # do nothing
        else:
            if opts.wandb_key == 'default':
                # taking the login API key from .netrc at user home dir
                opts.wandb_key = run_bash("awk '/api.wandb.ai/{getline; getline; print $2}' ~/.netrc")
            opts.wandb_login = f"wandb login {opts.wandb_key}"
        return opts


def is_commit_pushed(commit):
    branch = run_bash(f"git branch -r --contains {commit}")
    if branch == '':
        return False
    return True

def TBD(msg, *args, **kwargs):
    raise NotImplementedError(msg)

def FUTURE(msg, *args, **kwargs):
    """ Do nothing """
    pass

def yellow(msg):
    return(Style.BRIGHT + Fore.YELLOW + msg + Style.RESET_ALL)

def cyan(msg):
    return(Style.BRIGHT + Fore.CYAN + msg + Style.RESET_ALL)

def red(msg):
    return(Style.BRIGHT + Fore.LIGHTRED_EX + msg + Style.RESET_ALL)

def bold(msg):
    return(Style.BRIGHT + msg + Style.RESET_ALL)


def diff_to_latest_yaml(opts, experiment_name, experiment_yaml_fname):
    relevant_experiments = run_bash(
        f"ls {opts.sweep_history_dir} -lt | grep {experiment_name} | awk '{{print $NF}}' ").split('\n')
    relevant_experiments = [e for e in relevant_experiments if not e.startswith('uncommited')]
    if relevant_experiments:
        latest_relevant_experiment = relevant_experiments[0]
        if latest_relevant_experiment:
            latest_relevant_experiment = os.path.join(opts.sweep_history_dir, latest_relevant_experiment)

            numlines_above_diff = 2
            yaml_diff = run_bash(f"./utils/icdiff.py --cols=120 --numlines={numlines_above_diff}  {latest_relevant_experiment} {experiment_yaml_fname}")
            yaml_diff = remove_icdiff_text_below_match('submit_cmd', yaml_diff, numlines_above_diff+1)

            if yaml_diff:
                print(cyan(f'\n\nComparing to latest yaml ({opts.sweep_history_dir}/{latest_relevant_experiment}) :'))

                print(yaml_diff)
            else:
                print(cyan(f'\n\nWARNING: yaml is config is same as latest yaml ({opts.sweep_history_dir}/{latest_relevant_experiment}) '))

            # Path(temp_fname).unlink()  # delete temp file
    else:
        print(
            f'{bcolors.WARNING} No config file to compare to. This is first experiment for {bcolors.BOLD} {experiment_name} {bcolors.ENDC}')
        # + bcolors.OKGREEN + submit_cmd + bcolors.ENDC

def remove_icdiff_text_below_match(pattern, text, numlines_above_diff):
    import re
    lines = text.split('\n')
    edited = []
    for l in lines:
        if re.findall(pattern, l):
            break
        edited.append(l)
    return '\n'.join(edited[1:-numlines_above_diff])



def main(opts: SweepOpts):
    if opts.only_committed:
        if is_uncommited_git_repo(list_ignored_regex_pattern_filenames=opts.ignore_uncommited_dirs.split(','),
                                  ignore_untracked_files=True):
            raise RuntimeError('Uncommited files exist')

    if opts.git.clone_url:
        if not is_commit_pushed(opts.git.commit):
            raise RuntimeError(f'Commit {opts.git.commit} is not pushed to remote. You should first sync the local repo with the remote')



    # Load sweep YAML to read docker_image
    sweep_cfg = load_ordered_yaml(opts.sweep_yaml)
    sweep_name = alpha_numeric_uuid(length=3) + '_' + sweep_cfg['sweep_name']

    # Print sweep info
    print_sweep_info(sweep_cfg, sweep_name, countdown=30, opts=opts)

    FUTURE('test that the commit runs on a docker instance')

    # If code is based on uploading to workspace (rather than git clone), make the git archive and upload it to the workspace
    if opts.git.clone_url is None:
        # save the git commit to a zip archive under tmp_archive_dir
        tmp_archive_dir = f'/tmp/{opts.git.commit_archive_dir}'
        archive_fname = f"{tmp_archive_dir}/{opts.git.commit}.zip"
        run_bash(f"mkdir -p {tmp_archive_dir}; git archive -o {archive_fname} {opts.git.commit}")
        # upload to the workspace if run on NGC
        if opts.target.value == 'NGC':
            print(f'Uploading {archive_fname} to NGC workspace ...')
            upload_cmd = f"$HOME/ngc workspace upload --source {archive_fname} --destination {opts.git.commit_archive_dir} {opts.ngc.workspace}"
            upload_cmd = f"timeout -v {opts.ngc.upload_timeout} {upload_cmd}"
            print(upload_cmd)
            stdout, exit_code = run_bash(upload_cmd, return_exist_code=True, raise_on_err=False, )
            if exit_code != 0:
                    print(f'ERROR: Hit {opts.ngc.upload_timeout} sec timeout (or another error) while uploading code to the workspace. To resolve the issue, try to connect to another VPN, or to increase the upload_timeout limit on the YAML file.')
                    exit(exit_code)

            print(stdout)
            print('Done uploading')

    # Call W&B Sweep
    sweep_out = run_bash(f'wandb sweep {opts.sweep_yaml} --name {sweep_name} 2>&1')
    print(sweep_out)
    sweep_cmd = re.findall('wandb agent.*', sweep_out)[0]


    # prepare clone & checkout commands
    source_dir = opts.image_source_code_dir
    commit_depth = 1+int(run_bash(opts.git.cmd_commit_depth.format(commit=opts.git.commit)))
    cmd_clone = opts.git.cmd_clone.format(clone_url=opts.git.clone_url,
                                          commit_depth=commit_depth,
                                          source_dir=source_dir,
                                          commit=opts.git.commit,
                                          branch=opts.git.branch,
                                          )
    cmd_checkout = opts.git.cmd_checkout.format(
                                          source_dir=source_dir,
                                          commit=opts.git.commit,
                                          branch=opts.git.branch,
                                          )
    # prepare sweep agent command
    prefix_agent_commandline = dict(docker=opts.docker.prefix_agent_commandline_docker.format(source_dir=source_dir,
                                                                                              cmd_clone=cmd_clone,
                                                                 cmd_checkout=cmd_checkout,
                                                                 sweep_cmd=sweep_cmd, jupyter_cmd=opts.jupyter_cmd,
                                                                 webhost_cmd=opts.webhost_cmd,
                                                                 wandb_login=opts.wandb_login),
                                         NGC=opts.ngc.prefix_agent_commandline_ngc.format(source_dir=source_dir,
                                                                                          cmd_clone=cmd_clone,
                                                                 cmd_checkout=cmd_checkout,
                                                                 sweep_cmd=sweep_cmd, jupyter_cmd=opts.jupyter_cmd,
                                                                 webhost_cmd=opts.webhost_cmd,
                                                                 wandb_login=opts.wandb_login))[opts.target.value]

    sweep_agent_commandline = opts.experiment_commandline.format(source_dir=source_dir, cmd_clone=cmd_clone,
                                                                 cmd_checkout=cmd_checkout,
                                                                 prefix_agent_commandline=prefix_agent_commandline,
                                                                 sweep_cmd=sweep_cmd, jupyter_cmd=opts.jupyter_cmd,
                                                                 webhost_cmd=opts.webhost_cmd,
                                                                 wandb_login=opts.wandb_login)
    sweep_agent_commandline = sweep_agent_commandline.rstrip('\n')

    docker_image = sweep_cfg['docker_image']

    # prepare commands to submit the job
    sweep_hash = Path(sweep_cmd).stem
    commit_short = opts.git.commit[:3]
    job_name = opts.ngc.job_name.format(sweep_hash=sweep_name, commit=commit_short)

    run_cmd_docker = opts.docker.run_cmd_docker.format(source_dir=source_dir,
                                                       docker_image=docker_image, agent_cmd=sweep_agent_commandline,
                                                       commit_archive_dir=opts.git.commit_archive_dir)
    run_cmd_ngc = opts.ngc.run_cmd_ngc.format(ngc_instance=opts.ngc.ngc_instance, job_name=job_name,
                                          workspace=opts.ngc.workspace, docker_image=docker_image,
                                          agent_cmd=sweep_agent_commandline)

    if opts.target.value == 'docker':
        # submit locally
        print('Executing sweep with docker, using the following commandline:\n')
        command_colored = bcolors.BOLD + bcolors.OKGREEN + run_cmd_docker + bcolors.ENDC
        print(command_colored)
        run_bash(run_cmd_docker)
    elif opts.target.value == 'NGC':
        # submit to NGC
        num_machines = opts.ngc.num_machines
        print(f"\n\n{bcolors.UNDERLINE}Executing sweep on {num_machines} NGC machines, using the following commandline:{bcolors.ENDC}")
        command_colored = bcolors.BOLD + bcolors.OKGREEN + run_cmd_ngc + bcolors.ENDC
        print(command_colored)
        for _ in range(num_machines):
            run_bash(run_cmd_ngc)
    else:
        raise ValueError(opts.target.value)

    # log sweep to sweep_history
    # add uncommitted_files_list to yaml
    uncommitted_files_list = get_uncommitted_files([], ignore_untracked_files=True)
    log_sweep_history(sweep_cfg, sweep_name, commit_short, dict(submit_cmd_docker=run_cmd_docker, submit_cmd_ngc=run_cmd_ngc, uncommitted_files=uncommitted_files_list), opts)


def alpha_numeric_uuid(length):
    while True:
        uid = uuid.uuid4().hex[:length]
        if not uid.isnumeric():
            return uid



def print_sweep_info(sweep_cfg, sweep_name, opts: SweepOpts, countdown=30):
    swept_vars_str = '\n'.join(f'{key}: {values}' for key, values, in get_swept_vars(sweep_cfg).items())
    fixed_values_str = '\n'.join(f'{key}: {values}' for key, values, in get_fixed_values(sweep_cfg).items())
    num_combinations = len(list(itertools.product(*get_swept_vars(sweep_cfg).values())))

    print('\n'*100)  # clear the screen
    print(yellow('Setting:\n'), fixed_values_str, sep='')
    print(yellow(f'\nSweeping over:\n'), swept_vars_str, sep='')
    print(yellow(f'Number of combinations = {num_combinations}'))
    print(yellow(f'\nSweep method is: '), bold(sweep_cfg["method"] + ' search'))
    print(yellow('\nSweep name is: '), sweep_name, sep='')
    if opts.target.value == 'docker':
        print(yellow('\nTarget is:'), bold(f'{opts.target.value}'))
    if opts.target.value == 'NGC':
        print(yellow('\nTarget is:'), bold(f'{opts.ngc.num_machines}'), 'machines in', bold(opts.target.value))

    # visual diff to last yaml sweep
    diff_to_latest_yaml(opts, sweep_cfg['sweep_name'], opts.sweep_yaml)

    count_down(countdown,
               msg=red('\nWaiting for %d seconds. press ctrl-C to break, or any key to continue' % countdown),
               cont_on_any_key=True)


def log_sweep_history(sweep_cfg, sweep_hash, commit_short, submit_cmds, opts):
    # copy sweep yaml to sweep_history
    historical_yaml_fname = Path(opts.sweep_history_dir, f'{sweep_hash}_{commit_short}_{Path(opts.sweep_yaml).stem}.yaml')
    run_bash(f'cp {opts.sweep_yaml} {historical_yaml_fname}')
    # add run_cmd to yaml
    yaml_logging_txt = yaml.dump(dict(sweep_hash=sweep_hash, **submit_cmds), width=1e5)
    append_to_text_file(txt='\n' + yaml_logging_txt, fname=historical_yaml_fname)

    # add a two line summary to sweep_history.txt
    history_fname = f'{opts.sweep_history_dir}/sweep_history.txt'
    append_to_text_file(txt=f'# {historical_yaml_fname.stem} # sweeping over {get_swept_var_names(sweep_cfg)}', fname=history_fname)
    append_to_text_file(txt=get_fixed_values_query_string(sweep_cfg), fname=history_fname)


def get_swept_var_names(sweep_cfg):
    """ returns a string that represents the fixed hyper-params in a sweep. The string format is compatible
        with pandas dataframe queries"""
    var_names = []
    for key, dict_values in sweep_cfg['parameters'].items():
        if 'values' in dict_values and len(dict_values['values']) > 1:
            var_names.append(key)
    return var_names

def get_swept_vars(sweep_cfg):
    """ returns a string that represents the fixed hyper-params in a sweep. The string format is compatible
        with pandas dataframe queries"""
    swept_vars = {}
    for key, dict_values in sweep_cfg['parameters'].items():
        if 'values' in dict_values and len(dict_values['values']) > 1:
            swept_vars[key] = dict_values['values']
    return swept_vars


def get_fixed_values(sweep_cfg):
    """ returns a string that represents the fixed hyper-params in a sweep. The string format is compatible
        with pandas dataframe queries"""
    fixed_vars = {}
    for key, dict_values in sweep_cfg['parameters'].items():
        if 'value' in dict_values:
            value = dict_values['value']
        elif 'values' in dict_values and len(dict_values['values']) == 1:
            value = dict_values['values'][0]
        else:
            continue
        if isinstance(value, str):
            value = f'"{value}"'

        fixed_vars[key]=value

    return fixed_vars

def get_fixed_values_query_string(sweep_cfg):
    """ returns a string that represents the fixed hyper-params in a sweep. The string format is compatible
        with pandas dataframe queries"""

    return ' '.join(f'{key}=={values}' for key, values, in get_fixed_values(sweep_cfg).items())

def write_yaml_configuration(opts, experiment_yaml, experiment_name, target_fname=None, verbose=True):


    os.makedirs(opts.sweep_history_dir, exist_ok=True)

    uncommitted_files_list = get_uncommitted_files([], ignore_untracked_files=True)
    experiment_yaml['uncommitted_files'] = uncommitted_files_list

    if target_fname is None:
        target_fname = f'{experiment_name}%{experiment_yaml["queueing"]["timestamp"]}__{experiment_yaml["queueing"]["uid"]}.yaml'
        if uncommitted_files_list and not opts.args.ignore_uncommitted:
            target_fname = 'uncommited_' + target_fname

        target_fname = os.path.join(opts.sweep_history_dir, target_fname)
    with open(target_fname, 'w') as f:

        f.write(ruamel_yaml.dump(experiment_yaml, Dumper=ruamel_yaml.RoundTripDumper))

    if verbose:
        print('Saved experiment configuration under:\n', target_fname)

if __name__ == "__main__":
    _opts = SweepOpts.get_opts()

    main(_opts)

