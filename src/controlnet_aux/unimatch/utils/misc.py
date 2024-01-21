import os
import sys
import json


def read_text_lines(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)  # explicitly set exist_ok when multi-processing


def save_command(save_path, filename='command_train.txt'):
    check_path(save_path)
    command = sys.argv
    save_file = os.path.join(save_path, filename)
    # Save all training commands when resuming training
    with open(save_file, 'a') as f:
        f.write(' '.join(command))
        f.write('\n\n')


def save_args(args, filename='args.json'):
    args_dict = vars(args)
    check_path(args.checkpoint_dir)
    save_path = os.path.join(args.checkpoint_dir, filename)

    # save all training args when resuming training
    with open(save_path, 'a') as f:
        json.dump(args_dict, f, indent=4, sort_keys=False)
        f.write('\n\n')
