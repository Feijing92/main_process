import random as ran
import numpy as np
import os


def data_input(file_name, id_columns, label_column, head_line=True, dot=','):
  features, labels = [], []
  
  with open('./Kaggle_dataset/' + file_name + '.csv', 'r') as f:
    lines = f.readlines()
    if head_line:
      lines = lines[1:]

    print(file_name, len(lines))
    for line_num, line in enumerate(lines):
      line1 = line.strip('\n').split(dot)
      # print(line1)
      feature = []
      for column_num, num in enumerate(line1):
        if column_num not in id_columns:
          if column_num == label_column:
            labels.append(num)
          else:
            feature.append(num)
      features.append(feature)

  new_features = [[] for x in features]
  for i in range(len(features[0])):
    feature_values = list(set([x[i] for x in features]))
    xx = len(feature_values)
    print(i, xx)
    if xx > 100:
      new_values = continuous_feature(feature_values)
    else:
      new_values = discrete_feature(feature_values)
    
    for j, x in enumerate(features):
      new_features[j].append(new_values[x[i]])
  
  new_labels = []
  label_values = list(set(labels))
  new_values = discrete_feature(label_values)
  for i, x in enumerate(labels):
      new_labels.append(new_values[x])

  file_output(new_features, new_labels, file_name)
  print('*' * 40)


def discrete_feature(all_values):
  value2num = {}
  for i, value in enumerate(all_values):
    value2num[value] = i
  return value2num
  

def continuous_feature(all_values, divided_num=5):
  value2num = {}
  values = sorted(all_values)
  value_num = len(all_values)
  for i, value in enumerate(values):
    value2num[value] = int(divided_num * i / value_num)
  return value2num


def file_output(xs, ys, file_name):
  with open('./cleaned_dataset/'+file_name+'.csv', 'w') as f:
    feature_num = len(xs[0])
    f.write('id,')
    for i in range(1, feature_num + 1):
      f.write('feature_'+str(i) + ',')
    f.write('label\n')
    for i, data in enumerate(xs):
      # print(data)
      f.write(','.join([str(x) for x in data] + [str(ys[i])]) + '\n')


if __name__ == '__main__':
  data_input('airlines_delay', [0], 7)
  data_input('heart', [0,1], 13)
  data_input('income', [0], 14)
  data_input('network_domain', [0], 7)
  data_input('survey_lung_cancer', [], 15)
  data_input('Titanic', [0,3], 1)