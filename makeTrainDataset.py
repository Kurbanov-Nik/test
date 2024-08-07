import numpy as np
from sklearn.utils import shuffle

filesPath = "./out_2/prepared_data"

filesDerictoryNames = [
    "ru_exmpl",
    "ru_sum_num",
    "en_exmpl",
    "en_sum_num",
    "num_sum",
    "ru_en",
    "en_num"
]

train_X = np.load("%s/%s_data.npy" % (filesPath, filesDerictoryNames[0]), allow_pickle=True)
train_y = np.load("%s/%s_labels.npy" % (filesPath, filesDerictoryNames[0]), allow_pickle=True)

for directory in filesDerictoryNames[1:]:
    temp_X = np.load("%s/%s_data.npy" % (filesPath, directory), allow_pickle=True)
    temp_y = np.load("%s/%s_labels.npy" % (filesPath, directory), allow_pickle=True)
    train_X = np.append(train_X, temp_X, axis=0)
    train_y = np.append(train_y, temp_y, axis=0)

print("Test array with data have shape:", train_X.shape)
train_X, train_y = shuffle(train_X, train_y, random_state=15)
np.save("./out_2/train_X.npy", train_X)
np.save("./out_2/train_y.npy", train_y)