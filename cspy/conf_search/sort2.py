import pandas as pd

def main():

    df = pd.read_csv('energies.txt', sep=' ', header=None)

    df.columns=['conformer','energy']


    df = df.sort_values(by=['energy'])

    print(df)

    ordered_confs = df['conformer']

    ordered_confs.to_csv('reordered_confs.txt', index=None, header=False)

if __name__ == "__main__":
    main()