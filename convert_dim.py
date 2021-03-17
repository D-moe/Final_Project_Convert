from sklearn.decomposition import TruncatedSVD
import pandas as pd
import numpy as np

file_name= '/home/dmoe/Documents/CS263/FinalProject/GitHub/ECPE/data_combine/SBW-vectors-300-min5.txt'
#file_name = '/home/dmoe/Documents/CS263/FinalProject/GitHub/ECPE/ConvertEmbeddings/test.csv'
first_line = None
with open(file_name) as f:
    first_line = f.readline()
    f.close()
X = pd.read_csv(file_name, delimiter=r"\s+", header=None, skiprows=1).to_numpy()
print('Finished loading')
column_label = X[:,0]
to_transform = X[:,1:]
print(to_transform)
svd = TruncatedSVD(n_components = 200, n_iter = 7, random_state = 42)
print('Fitting')
svd.fit(to_transform)
print('Transforming')
to_transform = svd.transform(to_transform)
print('Insert column')
result = pd.DataFrame(to_transform)
result.insert(loc=0, column='0', value=column_label)
output_file = 'converted.csv'
with open(output_file, 'w') as out:
    out.write(first_line)
    out.close()
pd.DataFrame(result).to_csv('converted.csv', mode='a', header=False, index=False, sep=' ')
