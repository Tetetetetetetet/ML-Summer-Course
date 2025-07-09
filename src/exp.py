import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_rows', None)
data = pd.read_csv("Dataset/diabetic_data_training.csv")
print()
vc = data['medical_specialty'].value_counts()
# draw the bar chart
plt.bar(vc.index, vc.values)
plt.show()
