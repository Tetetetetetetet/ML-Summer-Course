"""
def categorize_diagnosis(new,old):
  '''
  The diagnosis columns are converted from icd9 codes to one of the 9 categories.
  The function creates a new column which consists of one of the 10 categories of disease.
  Nan values are considered as a new category
  '''
  #copying old column into new column
  data[new] = data[old]
  #filling NaN values with -1 category
  data[new] = data[new].fillna(-1)
  #if the code contians V or E then it is considered as category 0.
  data.loc[data[new].str.contains('V',na=False), [new]] = 0
  data.loc[data[new].str.contains('E',na=False), [new]] = 0
  #converting string column to float.
  data[new] = data[new].astype(float)
  #iterating through all the rows of the dataframe
  for index, row in data.iterrows():
      #checking if the code of the row belong to category 1,
      if (row[new] >= 390 and row[new] < 460) or (np.floor(row[new]) == 785):
          #assigning category 1 to the new column of the row.
          data.loc[index, new] = 1
      #checking if the code of the row belong to category 2,
      elif (row[new] >= 460 and row[new] < 520) or (np.floor(row[new]) == 786):
          #assigning category 2 to the new column of the row.
          data.loc[index, new] = 2
      elif (row[new] >= 520 and row[new] < 580) or (np.floor(row['new_diag1']) == 787):
          data.loc[index, new] = 3
      elif (np.floor(row[new]) == 250):
          data.loc[index, new] = 4
      elif (row[new] >= 800 and row[new] < 1000):
          data.loc[index, new] = 5
      elif (row[new] >= 710 and row[new] < 740):
          data.loc[index, new] = 6
      elif (row[new] >= 580 and row[new] < 630) or (np.floor(row[new]) == 788):
          data.loc[index, new] = 7
      elif (row[new] >= 140 and row[new] < 240):
          data.loc[index, new] = 8
      elif row[new] > 0 :
        #if the code of the row does not belong to any category then it is given 0.
          data.loc[index, new] = 0
"""

