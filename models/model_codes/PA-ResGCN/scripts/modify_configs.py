import os, argparse
os.chdir(os.getcwd())


parser = argparse.ArgumentParser(description='Modify Configs')
parser.add_argument('--path', type=str, default='', help='Your path to save preprocessed dataset')
parser.add_argument('--ntu60_path', type=str, default='', help='Your path to save NTU 60 dataset (S001 to S017)')
parser.add_argument('--ntu120_path', type=str, default='', help='Your path to save NTU 120 dataset (S018 to S032)')
args, _ = parser.parse_known_args()

for file in os.listdir('./configs'):
    with open(f'./configs/{file}', 'r') as f:
        lines = f.readlines()

    fr = open(f'./configs/{file}', 'w')
    for line in lines:
        if ' path:' in line:
            new_line = f'    path: {args.path}\n'
        elif ' ntu60_data_path:' in line:
            new_line = f'    ntu60_data_path: {args.ntu60_path}\n'
        elif ' ntu120_data_path:' in line:
            new_line = f'    ntu120_data_path: {args.ntu120_path}\n'
        else:
            new_line = line
        fr.write(new_line)
    fr.close()
