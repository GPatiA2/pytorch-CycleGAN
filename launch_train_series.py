import yaml
import os
import subprocess
import argparse

def options():

    parser = argparse.ArgumentParser(description='Launch training series')
    parser.add_argument('--trainings', type=str, default='training_params.yaml', help='Path to training parameters')
    args = parser.parse_args()
    print(args)
    return args

def launch_ind_training(yml_dict):

    training_args = []
    for arg_dict in yml_dict:
        for key, value in arg_dict.items():
            training_args.append(f'--{key}')
            if value is not None:
                training_args.append(f'{value}')

    print(training_args)
    subprocess.run(['python', 'train.py'] + training_args)

def main():

    args = options()
    with open(args.trainings, 'r') as file:
        yml_dict = yaml.load(file, Loader=yaml.FullLoader)

    for training_name in yml_dict:
        print("=============== NEW TRAINING ===============")
        print(f'{training_name}: {yml_dict[training_name]}')
        launch_ind_training(yml_dict[training_name])

    
if __name__ == '__main__':
    main()