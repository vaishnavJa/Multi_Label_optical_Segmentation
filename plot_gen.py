


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import sys


# In[14]:


param = sys.argv[1]



root = """E:/AI/FAPS/code/Mixedsupervision/results/{}/""".format(param)
loss_files = []
for path, subdirs, files in os.walk(root):
    for name in files:
        if name == "losses.csv":
            loss_files.append(os.path.join(path,name))

print(loss_files)

fig = plt.figure
for file in loss_files:
    inter = os.path.normpath(file)
    value = inter.split(os.sep)[7]
    df = pd.read_csv(file)
    print(value)
    print(df.tail(5))
    plt.plot(df['epoch'],df['validation_data'],label=value)
plt.title(param)
plt.legend()   
plt.savefig(os.path.join('.','plots',param+'.jpg'))
plt.show()


# In[27]:





# In[ ]:




