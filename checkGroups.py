import scipy.io

mat = scipy.io.loadmat('EILAT_data.mat')
data = mat["data"]
labels = data[0, 1][0]
div = data[0, 2]
dim1 = data[0, 3][0]
dim2 = data[0, 4][0]
counts = []
for group in div:
    tmp = []
    for i in range(min(labels), max(labels) + 1):
        tmp.append(0)
    for ind in group:
        tmp[labels[ind - 1] - 1] += 1
    counts.append(tmp)

diffs = []
for i in range(min(labels), max(labels) + 1):
    for j in range(len(div)):
        diffs.append(abs(counts[j][i - 1] - counts[0][i - 1]))
print(max(diffs))