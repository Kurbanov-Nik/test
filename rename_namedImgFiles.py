import os
import pandas as pd

gt = pd.DataFrame(columns=['file_name', 'value'])

# filePath = "meshups/en_num"
filePath = "en_examples"

indx = 0
for fileElem in os.listdir("./%s" % filePath):
    if fileElem != "labels.txt":
        row = (fileElem[:fileElem.rfind('.')]) # name_1111.jpg -> name_1111 -[.jpg]
        gt.loc[gt.shape[0]] = ("%d.jpg" % indx, row.split("_")[0]) # name_1111 -> name -[_1111]
        os.rename("./%s/%s.jpg" % (filePath, row), "./%s/%d.jpg" % (filePath, indx))
        indx += 1

with open("./%s/%s.txt" % (filePath, "labels"), 'w') as file:
    for index, row in gt.iterrows():
        file.write(f"{row['file_name']} {row['value']}\n")