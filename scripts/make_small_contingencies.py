import pandas as pd

def make_small():
    df = pd.read_csv('data/contingencies/contingencies_1354.csv')
    small = df.head(200)
    small.to_csv('data/contingencies/contingencies_1354_small.csv', index=False)
    print('Wrote', len(small), 'contingencies to data/contingencies/contingencies_1354_small.csv')

if __name__ == '__main__':
    make_small()
