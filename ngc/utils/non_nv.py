import subprocess
import sys

def run_bash(cmd, raise_on_err=True, raise_on_warning=False, versbose=True, return_exist_code=False, err_ind_by_exitcode=False):
    """
    Source:  https://github.com/yuvalatzmon/COSMO/
    This function takes a Bash command and return its stdout
    Returns: string (stdout)
    :type cmd: string
    """
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, executable='/bin/bash')
    # p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    out, err = p.communicate()
    out = out.strip().decode('utf-8')
    err = err.strip().decode('utf-8')
    exit_code = p.returncode
    is_err = err != ''
    if err_ind_by_exitcode:
        is_err = (exit_code != 0)

    if is_err and raise_on_err:
        do_raise = True
        if 'warning' in err.lower():
            do_raise = raise_on_warning
            if versbose and not raise_on_warning:
                print('command was: {}'.format(cmd))
            print(err, file=sys.stderr)

        if do_raise or 'error' in err.lower():
            if versbose:
                print('command was: {}'.format(cmd))
            raise RuntimeError(err)

    if return_exist_code:
        return out, exit_code
    else:
        return out  # This is the stdout from the shell command


