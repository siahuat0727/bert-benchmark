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


# TODO no need split
def get_method_name(f):
    return f.split('.')[0].split('#')[1]


def get_batch_size(f):
    return f.split('.')[0].split('#')[2]


def get_repeat_number(f):
    return f.split('.')[0].split('#')[3]

def get_model_name(f):
    return f.split('#')[4]

def get_seq_len(f):
    return f.split('#')[5]

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

        skip = False
        for i, row in df.iterrows():
            if str(row['Type']) == 'Type':
                skip = True
            if skip:
                df.drop(i, inplace=True)
                continue

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

        # TODO no need this?
        df = df.dropna()

        df = df[df['Type'].str.contains('GPU activities')]
        df = df[~df['Name'].str.contains('memcpy')]
        df = df[~df['Name'].str.contains('memset')]

        df = df.drop(['Type', 'Name'], axis=1)

        repeat_time = int(get_repeat_number(f))
        number = 10  # TODO no magic number

        total_iteration = repeat_time * number
        if get_method_name(f) == 'nnfusion':
            total_iteration = 105

        df['Time'] = df['Time'].astype(float).div(total_iteration)
        total_time = sum(float(v) for v in df['Time'])

        df = df.drop(['Time'], axis=1)
        df = df.drop_duplicates()
        df['result'] = total_time

        df['model'] = get_model_name(f)
        df['sequence_length'] = get_seq_len(f)

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

    df = df.sort_values(by='batch_size', key=pd.to_numeric)
    df = df.sort_values(by='methods', kind='stable')
    print(df)

    plt.figure(figsize=(10, 6))

    ax = sns.barplot(x='batch_size', y='result', hue='methods', data=df)
    ax.set_title(title)
    ax.set_ylabel(infos['ylabel'])

    img_path = f"inference-{infos['type']}-{frmt}.png"
    print(f'Save {img_path}')
    plt.savefig(img_path)


def plot_error(files):
    def get_runtime_and_value(f):
        runtime = f.split('_')[-1].split('.')[0]
        with open(f) as f:
            max_abs_err = f.readline().strip()
        return {
            'runtime': runtime,
            'max_abs_error': max_abs_err,
        }
    data = [get_runtime_and_value(f)
        for f in files]
    df = pd.DataFrame.from_records(data)
    df = df.sort_values(by='runtime', kind='stable')
    df['max_abs_error'] = pd.to_numeric(df['max_abs_error'], errors='coerce')
    df = df.dropna()
    print(df)

    ax = sns.barplot(x='runtime', y='max_abs_error', hue='runtime', data=df)
    ax.set_title('Max absolute error of each runtime')
    ax.set_yscale("log")

    plt.xticks(rotation=-15)
    # ax.set_ylabel(infos['ylabel'])
    ax.set_xlabel('runtimes')

    img_path = "inference-max-abs-error.png"
    print(f'Save {img_path}')
    plt.savefig(img_path)


def main():
    parser = ArgumentParser()
    parser.add_argument('--files', nargs='+', help="Files to be plotted")
    parser.add_argument('--type', default='speed',
                        choices=['speed', 'memory', 'error'], help="")
    parser.add_argument('--frmt', default='python',
                        choices=['python', 'nvprof'], help="")
    args = parser.parse_args()
    if not args.files:
        raise AssertionError('Files are needed to be plotted')
    infos = get_infos(args)
    if args.type == 'error':
        plot_error(args.files)
    else:
        plot(args.files, infos, args.frmt)


main()
