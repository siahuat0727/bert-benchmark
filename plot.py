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

def extract_nvprof_df_from_csv(files):
    def do_extract_nvprof_df_from_csv(f):
        df = pd.read_csv(f).assign(
                 methods=get_method_name(f),
                 batch_size=get_batch_size(f),
             )
        df = df.drop(['Time(%)', 'Calls', 'Avg', 'Min', 'Max', 'Type'], axis=1)

        unit = df['Time'].iloc[0]
        print(unit)
        df = df.drop(0)

        df = df[~df['Name'].str.contains('memcpy')]
        df = df[~df['Name'].str.contains('memset')]
        df = df.drop(['Name'], axis=1)

        total_time = sum(float(v) for v in df['Time'])
        assert unit in ('ms', 'us')
        if unit == 'us':
            total_time *= 1000

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
        'hugging': extract_hugging_df_from_csv,
    }[frmt](files)

    # TODO Make sure all the eles in col is the same
    model, seq_len = df['model'].iloc[0], df['sequence_length'].iloc[0]
    df = df.drop(['model', 'sequence_length'], axis=1)

    title = f"Inference {infos['type']} test for {model} with {seq_len} seq. length"

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
    parser.add_argument('--frmt', default='hugging',
                        choices=['hugging', 'nvprof'], help="")
    args = parser.parse_args()
    assert args.files, 'Files are needed to be plotted'
    infos = get_infos(args)
    plot(args.files, infos, args.frmt)


main()
