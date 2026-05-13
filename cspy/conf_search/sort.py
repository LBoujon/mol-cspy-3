import pandas as pd

def main():
    df = pd.read_csv('g09_engs', sep=' ', header=None)

    df.columns=['conformer','energy_Ha']


    df = df.sort_values(by=['energy_Ha'])
    df['conformer'] = df['conformer']+'.xyz'

    print(df)

    #df.to_csv('g09_engs', index=None, header=False, sep=' ')

    ordered_confs = df['conformer']


    ordered_confs.to_csv('reordered_confs', index=None, header=False)

if __name__ == "__main__":
    main()