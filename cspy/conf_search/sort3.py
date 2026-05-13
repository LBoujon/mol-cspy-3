import pandas as pd


def main():
    df = pd.read_csv("conf_areas.csv")

    df = df.sort_values(by=['energy'])

    columns = df['conformer']

    columns.to_csv("reordered_confs", index=None, header=None)

if __name__ == "__main__":
    main()