import datetime
import os
import pickle
import subprocess
import sys


def print_args(args):
    opts = vars(args)
    print('======= Options ========')
    for k, v in sorted(opts.items()):
        print('{}: {}'.format(k, v))
    print('========================')


def save_args(args, save_folder, opt_prefix='opt', verbose=True):
    opts = vars(args)
    # Create checkpoint folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)

    # Save options
    opt_filename = '{}.txt'.format(opt_prefix)
    opt_path = os.path.join(save_folder, opt_filename)
    with open(opt_path, 'a') as opt_file:
        opt_file.write('====== Options ======\n')
        for k, v in sorted(opts.items()):
            opt_file.write(
                '{option}: {value}\n'.format(option=str(k), value=str(v)))
        opt_file.write('=====================\n')
        opt_file.write('launched {} at {}\n'.format(
            str(sys.argv[0]), str(datetime.datetime.now())))

        # Add git info
        label = subprocess.check_output(["git", "describe",
                                         "--always"]).strip()
        if subprocess.call(
            ["git", "branch"],
                stderr=subprocess.STDOUT,
                stdout=open(os.devnull, 'w')) == 0:
            opt_file.write('=== Git info ====\n')
            opt_file.write('{}\n'.format(label))
            commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
            opt_file.write('commit : {}\n'.format(commit.strip()))

    opt_picklename = '{}.pkl'.format(opt_prefix)
    opt_picklepath = os.path.join(save_folder, opt_picklename)
    with open(opt_picklepath, 'wb') as opt_file:
        pickle.dump(opts, opt_file)
    if verbose:
        print('Saved options to {}'.format(opt_path))
