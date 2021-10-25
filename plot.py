from argparse import ArgumentParser

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def get_infos(args):
    if args.type == 'speed':
        return {
            'type': args.type,
            'extract_method': lambda x: x[6:-4],  # speed-<method>.csv
            'ylabel': 'inference-time (s)',
        }
    return {
        'type': args.type,
        'extract_method': lambda x: x[7:-4],   # memory-<method>.csv
        'ylabel': 'inference-memory (MB)',
    }


def plot(files, infos):
    df = pd.concat([
        pd.read_csv(f).assign(methods=infos['extract_method'](f))
        for f in files
    ])

    # TODO Make sure all the eles in col is the same
    model, seq_len = df['model'].iloc[0], df['sequence_length'].iloc[0]
    df = df.drop(['model', 'sequence_length'], axis=1)

    title = f"Inference {infos['type']} test for {model} with {seq_len} seq. length"

    print(df)

    plt.figure(figsize=(10, 6))

    ax = sns.barplot(x='batch_size', y='result', hue='methods', data=df)
    ax.set_title(title)
    ax.set_ylabel(infos['ylabel'])

    plt.savefig(f"inference-{infos['type']}.png")


def main():
    parser = ArgumentParser()
    parser.add_argument('--files', nargs='+', help="Files to be plotted")
    parser.add_argument('--type', default='speed',
                        choices=['speed', 'memory'], help="")
    args = parser.parse_args()
    assert args.files, 'Files are needed to be plotted'
    infos = get_infos(args)
    plot(args.files, infos)


main()
