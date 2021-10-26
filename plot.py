from argparse import ArgumentParser

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def get_infos(args):
    if args.type == 'speed':
        return {
            'type': args.type,
            'ylabel': 'inference-time (s)',
        }
    return {
        'type': args.type,
        'ylabel': 'inference-memory (MB)',
    }


def get_method_name(f):
    return f.split('.')[0].split('#')[1]


def get_batch_size(f):
    return f.split('.')[0].split('#')[2]

def is_float(s):
    try:
        float(s)
    except Exception as e:
        return False
    return True

def extract_nvprof_df_from_csv(files):
    def do_extract_nvprof_df_from_csv(f):
        df = pd.read_csv(f).assign(
                 methods=get_method_name(f),
                 batch_size=get_batch_size(f),
             )
        df = df.drop(['Time(%)', 'Calls', 'Avg', 'Min', 'Max'], axis=1)

        unit = df['Time'].iloc[0]
        for i, row in df.iterrows():
            if str(row['Time']) in ('s, ms, us'):
                unit = row['Time']
            if is_float(row['Time']):
                t = float(row['Time'])
                if unit == 'us':
                    t /= 10**6
                elif unit == 'ms':
                    t /= 10**3
                df.at[i, 'Time'] = str(t)

        df = df.drop(0)

        df = df.dropna()

        df = df[df['Type'].str.contains('GPU activities')]
        df = df[~df['Name'].str.contains('memcpy')]
        df = df[~df['Name'].str.contains('memset')]

        df = df.drop(['Type', 'Name'], axis=1)
        total_time = sum(float(v) for v in df['Time'])

        df = df.drop(['Time'], axis=1)
        df = df.drop_duplicates()
        df['result'] = total_time

        # TODO
        df['model'] = 'BERT-based-cased'
        df['sequence_length'] = 512

        return df

    return pd.concat([
        do_extract_nvprof_df_from_csv(f)
        for f in files
    ])

def extract_hugging_df_from_csv(files):
    return pd.concat([
        pd.read_csv(f).assign(methods=get_method_name(f))
        for f in files
    ])


def plot(files, infos, frmt=''):
    df = {
        'nvprof': extract_nvprof_df_from_csv,
        'python': extract_hugging_df_from_csv,
    }[frmt](files)

    # TODO Make sure all the eles in col is the same
    model, seq_len = df['model'].iloc[0], df['sequence_length'].iloc[0]
    df = df.drop(['model', 'sequence_length'], axis=1)

    title = f"Inference {infos['type']} test for {model} with {seq_len} seq. length"

    df = df.sort_values(by='batch_size', key=lambda col: pd.to_numeric(col))
    df = df.sort_values(by='methods', kind='stable')
    print(df)

    plt.figure(figsize=(10, 6))

    ax = sns.barplot(x='batch_size', y='result', hue='methods', data=df)
    ax.set_title(title)
    ax.set_ylabel(infos['ylabel'])

    img_path = f"inference-{infos['type']}-{frmt}.png"
    print(f'Save {img_path}')
    plt.savefig(img_path)


def main():
    parser = ArgumentParser()
    parser.add_argument('--files', nargs='+', help="Files to be plotted")
    parser.add_argument('--type', default='speed',
                        choices=['speed', 'memory'], help="")
    parser.add_argument('--frmt', default='python',
                        choices=['python', 'nvprof'], help="")
    args = parser.parse_args()
    assert args.files, 'Files are needed to be plotted'
    infos = get_infos(args)
    plot(args.files, infos, args.frmt)


main()
