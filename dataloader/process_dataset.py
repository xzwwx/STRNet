import os

with open('/data/wucong/something-something-v1_list/something-something-v1-labels.csv') as f:
    lines = f.readlines()
categories = []
for line in lines:
    line = line.rstrip()
    categories.append(line)
categories = sorted(categories)
with open('/data/wucong/something-something-v1_list/category.txt', 'w') as f:
    f.write('\n'.join(categories))

dict_categories = {}
for i, category in enumerate(categories):
    dict_categories[category] = i

files_input = ['/data/wucong/something-something-v1_list/something-something-v1-validation.csv', '/data/wucong/something-something-v1_list/something-something-v1-train.csv']
files_output = ['/data/wucong/something-something-v1_list/val_videofolder.txt', '/data/wucong/something-something-v1_list/train_videofolder.txt']
for (filename_input, filename_output) in zip(files_input, files_output):
    with open(filename_input) as f:
        lines = f.readlines()
    folders = []
    idx_categories = []
    for line in lines:
        line = line.rstrip()
        items = line.split(';')
        folders.append(items[0])
        idx_categories.append(os.path.join(str(dict_categories[items[1]])))
    output = []
    for i in range(len(folders)):
        curFolder = folders[i]
        curIDX = idx_categories[i]
        dir_files = os.listdir(os.path.join('/data/wucong/something-something-v1', curFolder))
        output.append('%s %d %d' % (curFolder, len(dir_files), int(curIDX)))
    with open(filename_output, 'w') as f:
        f.write('\n'.join(output))
