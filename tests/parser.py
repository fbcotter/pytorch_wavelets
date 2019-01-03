# coding: utf-8
import pandas as pd
import numpy as np
import argparse
from io import StringIO
parser = argparse.ArgumentParser(description='Prof parser')
parser.add_argument('file', type=str)

# Strip the whitespace on reading
def strip(text):
    try:
        return text.strip()
    except AttributeError:
        return text

def convert(x):
    if x[-2:] == 'ms':
        x = float(x[:-2])
    elif x[-2:] == 'us':
        x = 0.001 * float(x[:-2])
    elif x[-2:] == 'ns':
        x = 1e-6 * float(x[:-2])
    elif x[-1] == 's':
        x = 1000 * float(x[:-1])
    return x

def convert_timep(x):
    return float(x[:-1])


def pandify(data):
    names = ['Type', 'Time (%)', 'Time (ms)', 'Calls', 'Avg', 'Min', 'Max',
             'Name']
    convs = {i: strip for i in range(8)}
    df = pd.read_csv(StringIO(data), names=names, skiprows=0, sep=';',
                     converters=convs)
    df.loc[df.loc[:,'Type'] == '', 'Type'] = np.NaN
    df = df.fillna(method='ffill')
    df['Time (ms)'] = df['Time (ms)'].apply(convert)
    df['Time (%)'] = df['Time (%)'].apply(convert_timep)
    df['Calls'] = df['Calls'].apply(int)
    df1 = df[df['Type'] == 'GPU activities:']
    df2 = df[df['Type'] == 'API calls:']
    idx = df1.Name.str.contains('CUDA memcpy')
    s1 = pd.Series({'Type': 'Total:',
                    'Time (%)': df1['Time (%)'].sum(),
                    'Time (ms)': df1['Time (ms)'].sum(),
                    'Calls': df1['Calls'].sum(),
                    'Avg': '', 'Min': '', 'Max': '', 'Name': ''})
    s2 = pd.Series({'Type': 'Total (no mem):',
                    'Time (%)': df1.loc[~idx, 'Time (%)'].sum(),
                    'Time (ms)': df1.loc[~idx, 'Time (ms)'].sum(),
                    'Calls': df1.loc[~idx, 'Calls'].sum(),
                    'Avg': '', 'Min': '', 'Max': '', 'Name': ''})
    s3 = pd.Series({'Type': 'Total:',
                    'Time (%)': df2['Time (%)'].sum(),
                    'Time (ms)': df2['Time (ms)'].sum(),
                    'Calls': df2['Calls'].sum(),
                    'Avg': '', 'Min': '', 'Max': '', 'Name': ''})

    df3 = pd.concat([df1, pd.concat([s1, s2], axis=1).T,
                     df2, pd.concat([s3,], axis=1).T],
                    ignore_index=True, axis=0)
    return df3


def prep_file(file):
    with open(file, 'r') as f:
        data = f.readlines()

    header = data[0].split('command: ')[1]
    for i, l in enumerate(data):
        if i >= 3:
            data[i] = ';'.join([l[:16], l[17:25], l[26:35], l[36:45],
                              l[46:55], l[56:65], l[66:75], l[76:]])
    data = ''.join(data[4:])
    return data, header

if __name__ == '__main__':
    args = parser.parse_args()
    data, header = prep_file(args.file)
    df = pandify(data)
    with open(args.file, 'w') as f:
        f.write(header)
        f.write(df.to_string(index=False))

