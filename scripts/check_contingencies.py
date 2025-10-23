with open('data/contingencies/contingencies_1354.txt', 'r') as f:
    for i, line in enumerate(f):
        print(line.strip())
        if i >= 4:
            break