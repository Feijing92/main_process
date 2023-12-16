import random as ran
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.inset_locator import inset_axes
from matplotlib.lines import Line2D
from sklearn.metrics import average_precision_score, roc_curve, auc
import pickle
from scipy.special import comb
from functools import partial
from pathos.pools import ProcessPool, ThreadPool
from tqdm import tqdm
import time
import math
import os



def parallel(func, *args, show=False, thread=False, **kwargs):

    p_func = partial(func, **kwargs)
    pool = ThreadPool() if thread else ProcessPool()
    try:
        if show:
            start = time.time()
            with tqdm(total=len(args[0]), desc="computing process") as t:
                r = []
                for i in pool.imap(p_func, *args):
                    r.append(i)
                    t.set_postfix({'parallel function': func.__name__, "computing cost": "%ds" % (time.time() - start)})
                    t.update()
        else:
            r = pool.map(p_func, *args)
        return r
    except Exception as e:
        print(e)
    finally:
        pool.close()  # close the pool to any new jobs
        pool.join()  # cleanup the closed worker processes
        pool.clear()  # Remove server with matching state


class Data:

  def __init__(self, file_path, file_name, division):
    self.file_path = file_path
    self.file_name = file_name
    self.division = division

  def dataset_input(self):
    self.data = []
    self.label = []

    with open(self.file_path) as f:
      lines = f.readlines()
      for line in lines:
        new_line = line.strip('\n').split(',')
        # print(new_line)
        self.data.append([int(x) for x in new_line[:-2]])
        self.label.append(int(new_line[-2]))
    
    self.all_features = []
    for x in self.data:
      if x not in self.all_features:
        self.all_features.append(x)
    
    self.original_feature_distribution = dict(zip(list(range(len(self.all_features))), [[0, 0] for x in self.all_features]))
    for i, x in enumerate(self.data):
      xi = self.all_features.index(x)
      # print(i, self.label[i])
      self.original_feature_distribution[xi][self.label[i]] += 1

  def dataset_division(self):
    self.dataset_input()
    print('input is ready!')
    self.divided_feature_distribution = dict(zip(list(range(len(self.all_features))), [[0, 0, 0, 0] for x in self.all_features]))
    self.xtrain, self.ytrain, self.xtest, self.ytest = [], [], [], []
    for key, value in self.original_feature_distribution.items():
      p1, n1 = value

      for i in range(p1):
        if ran.random() < self.division:
          self.divided_feature_distribution[key][1] += 1
          self.xtrain.append(self.all_features[key])
          self.ytrain.append(1)
        else:
          self.divided_feature_distribution[key][3] += 1
          self.xtest.append(self.all_features[key])
          self.ytest.append(1)
      for i in range(n1):
        if ran.random() < self.division:
          self.divided_feature_distribution[key][0] += 1
          self.xtrain.append(self.all_features[key])
          self.ytrain.append(0)
        else:
          self.divided_feature_distribution[key][2] += 1
          self.xtest.append(self.all_features[key])
          self.ytest.append(0)

    if len(self.xtest) == 0:
      self.xtest = self.xtrain
      self.ytest = self.ytrain

  def feature_construction(self):
    self.dataset_division()
    def optimal_score(x):
      xi = self.all_features.index(x)
      value = self.divided_feature_distribution[xi]
      value_sum = value[0] + value[1]
      if value_sum > 0:
        return 1.0*value[1] / value_sum
      else:
        return 0.5
    return [optimal_score(x) for x in self.xtrain], [optimal_score(x) for x in self.xtest]
  
  def theoretical_error(self):
    self.dataset_input()
    all_p = [round(0.1*i, 1) for i in range(11)]
    if len(self.data) > 6000:
      sampling = 20
    else:
      sampling = 200

    m = len(self.data)
    theoretical_results = []
    for p in all_p:
      result = [0, 0, 0]
      for turn in range(sampling):
        a = 0
        k1, k2, k3 = 0, 0, 0
        for value in self.original_feature_distribution.values():
          total_n, total_p = value
          a1, b1 = 0, 0
          for i in range(total_p):
            if ran.random() < p:
              a1 += 1
          for i in range(total_n):
            if ran.random() < p:
              b1 += 1
          a += a1 + b1
          if p > all_p[0]:
            k1 += min(a1, b1)
          if p < all_p[-1]:
            k2 += max(total_p-a1, total_n-b1)
          k3 += (max(a1, b1) + max(total_p-a1, total_n-b1) - max(total_n, total_p))
        if p > all_p[0]:
          result[0] += k1 / a
        if p < all_p[-1]:
          result[1] += k2 / (m - a)
        result[2] += k3 / m
      theoretical_results.append([r / sampling for r in result])
    
    simulation_results = []
    for p in all_p:
      print(p)
      self.division = p
      result = [0, 0, 0]
      for turn in range(sampling):
        self.dataset_division()
        if p > all_p[0]:
          result[0] += self.min_hinge_loss() / (m*p)
        if p < all_p[-1]:
          result[1] += self.max_accuracy() / (m*(1-p))
        result[2] += self.error_bound() / m
      simulation_results.append([r / sampling for r in result])

    return [theoretical_results, simulation_results]
  
  def min_hinge_loss(self):
    return sum([min(value[0], value[1]) for value in self.divided_feature_distribution.values()])

  def max_accuracy(self, only_train=False):
    if only_train:
      return sum([max(value[0], value[1]) for value in self.divided_feature_distribution.values()])
    else:
      return sum([max(value[2], value[3]) for value in self.divided_feature_distribution.values()])
  
  def NN(self):
    data = np.array(self.xtrain)
    label = np.array(self.ytrain)
    clf = MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(100,100,100), random_state=1,max_iter=max_training_turn)
    clf.fit(data, label)
    y_pred1 = clf.predict(self.xtrain)
    y_pred2 = clf.predict(self.xtest)
    return y_pred1, y_pred2
  
  def error_bound(self):
    return sum([min(- value[3] + value[0], - value[2] + value[1]) + max(value[3], value[2]) - min(value[0], value[1]) for value in self.divided_feature_distribution.values()])

  def training_and_prediction(self, training_turn_max):
    optimal_pred = self.feature_construction()

    x1, y1 = len(self.xtrain), len(self.xtest)
    n_positive = sum(self.label)
    n_negative = len(self.label) - n_positive
    features_num = len(self.data[0])
    print(self.file_name+':', x1, y1, n_positive, n_negative, features_num, self.division)

    error_bound_num = self.error_bound()
    min_hinge = self.min_hinge_loss()
    
    if self.division < 1:
      max_accuracy = self.max_accuracy()
    else:
      max_accuracy = self.max_accuracy(only_train=True)

    output_results = [error_bound_num, x1, y1, min_hinge, max_accuracy]
    all_train_pred = [self.ytrain]
    all_test_pred = [self.ytest]

    # XGBoost
    dtrain = xgb.DMatrix(np.array(self.xtrain), label=np.array(self.ytrain))
    dtest = xgb.DMatrix(self.xtest)
    for obj in ['reg:squarederror', 'reg:logistic', 'binary:hinge']:
      params = {'booster': 'gbtree', 'objective': obj}
      model = xgb.train(params, dtrain, training_turn_max)
      train_pred = model.predict(dtrain)
      test_pred = model.predict(dtest)
      all_train_pred.append(train_pred)
      all_test_pred.append(test_pred)
    
    model = xgb.train({'booster': 'gbtree', 'objective': 'multi:softmax', 'num_class': 2}, dtrain, training_turn_max)
    train_pred = model.predict(dtrain)
    test_pred = model.predict(dtest)
    all_train_pred.append(train_pred)
    all_test_pred.append(test_pred)

    # MLP
    y_pred = self.NN()
    all_train_pred.append(y_pred[0])
    all_test_pred.append(y_pred[1])

    # optimal
    all_train_pred.append(optimal_pred[0])
    all_test_pred.append(optimal_pred[1])
  
    return output_results, all_train_pred, all_test_pred


def continuous_classifier_hinge_accuracy(y_train, train_pred, y_test, test_pred):
  score_distribution = {}
  for i, x in enumerate(train_pred):
    if x not in score_distribution:
      score_distribution[x] = [0,0,0,0]
    score_distribution[x][y_train[i]] += 1

  for i, x in enumerate(test_pred):
    if x not in score_distribution:
      score_distribution[x] = [0,0,0,0]
    score_distribution[x][y_test[i] + 2] += 1  
  
  all_scores = sorted(list(score_distribution.keys()), reverse=True)
  hinge_loss = sum([value[1]  for value in score_distribution.values()])
  accuracy = sum([value[2]  for value in score_distribution.values()])
  score_pairs = [[hinge_loss, accuracy]]
  for score in all_scores:
    value = score_distribution[score]
    hinge_loss += value[0] - value[1]
    accuracy += value[3] - value[2]
    score_pairs.append([hinge_loss, accuracy])

  sorted_score_pairs = sorted(score_pairs, key=lambda x: x[0])

  return sorted_score_pairs[0]


def discrete_classifier_hinge_accuracy(y_train, train_pred, y_test, test_pred):
  accuracy_score = 0
  hinge_loss = 0
  for i, x in enumerate(train_pred):
    if x != y_train[i]:
      hinge_loss += 1
  
  for i, x in enumerate(test_pred):
    if x == y_test[i]:
      accuracy_score += 1
  
  return [hinge_loss, accuracy_score]


def experiment(file, pp, ix, t):
  dataset = Data('./cleaned_dataset/' + file, file, pp)
  result = dataset.training_and_prediction(t)  
  basics, train_preds, test_preds = result

  n, max_acc = basics[1], basics[-1]
  print(file, pp, ix)
  print('max acc:', max_acc/n)
  with open(output_document + '/' + file + '_' + str(pp) + '_' + str(ix) + '_' + str(t) + '.txt', 'wb') as f:
    pickle.dump(result, f)


def uc_experiment(file, pp, ix, t):
  dataset = Data('./cleaned_UC_dataset/' + file, file, pp)
  result = dataset.training_and_prediction(t)  
  basics, train_preds, test_preds = result
  n, max_acc = basics[1], basics[-1]
  print(file, pp, ix)
  print('max acc:', max_acc/n)
  y_label = train_preds[0]
  for j, y in enumerate(train_preds[1:]):
    fpr1, tpr1, thresholds1 = roc_curve(y_label, y, pos_label=1)
    roc_auc = auc(fpr1, tpr1)
    print(roc_auc)
  print('*' * 40)
  
  with open(output_document + '/' + file + '_' + str(pp) + '_' + str(ix) + '_' + str(t) + '.txt', 'wb') as f:
    pickle.dump(result, f)


def figure1(files, h1, h2, et):
  fig = plt.figure()
  plt.subplots_adjust(left=0.05,bottom=0.05,top=0.9,right=0.95,hspace=0.25,wspace=0.2)
  palette = plt.get_cmap('Set3')
  plt.rcParams['xtick.direction'] = 'in'
  plt.rcParams['ytick.direction'] = 'in'
  aucs = [[] for file in files]
  accuracys = [[] for file in files]

  for i, file in enumerate(files):
    with open(output_document + '/' + file + '_1_0_' + str(et)+ '.txt', 'rb') as f1:
      result = pickle.load(f1)
    
    basics, train_preds, test_preds = result[:3]
    ax = plt.subplot(h1,h2,i+1)
    y_label = train_preds[0]
    for j, y in enumerate(train_preds[1:]):
      fpr1, tpr1, thresholds1 = roc_curve(y_label, y, pos_label=1)
      roc_auc = auc(fpr1, tpr1)
      if j < 2 or j == 5:
        acc = continuous_classifier_hinge_accuracy(y_label, y, y_label, y)[1] / len(y_label)
      else:
        acc = discrete_classifier_hinge_accuracy(y_label, y, y_label, y)[1] / len(y_label)
      print(file, method_names[j], 'auc:', roc_auc, 'ac:', acc)
      if i == 0:
        plt.plot(fpr1, tpr1, color=palette(color_indices[j]), linewidth=3, alpha=0.9, label=method_names[j])
      else:
        plt.plot(fpr1, tpr1, color=palette(color_indices[j]), linewidth=3, alpha=0.9)
      
      accuracys[i].append(acc)
      aucs[i].append(roc_auc)
    
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xticks([-0.02, 0.2, 0.4, 0.6, 0.8, 1.02])
    ax.set_yticks([-0.02, 0.2, 0.4, 0.6, 0.8, 1.02])
    if i in [2,3]:
      ax.set_xticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'], weight='bold', fontsize=12,fontproperties='Times New Roman')
    else:
      ax.xaxis.set_major_formatter(plt.NullFormatter())
    if i in [0,2]:
      ax.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'], weight='bold', fontsize=12,fontproperties='Times New Roman')
    else:
      ax.yaxis.set_major_formatter(plt.NullFormatter())

    ax.text(0.95, 0.15, data_names[i], horizontalalignment='right', verticalalignment='top', transform=ax.transAxes, weight='bold', fontsize=16,fontproperties='Times New Roman')
    ax.text(-0.1, 1.025, subfigure_index[i], horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes, weight='bold', fontsize=16,fontproperties='Times New Roman')

  fig.legend(bbox_to_anchor=(0.5,-0.1), loc='upper center', frameon=False, ncol=3,prop=font1)
  fig.text(0.5, -0.05, 'False Positive Rate', ha='center', fontsize=16,weight='bold',fontproperties='Times New Roman')
  fig.text(-0.05, 0.5, 'True Positive Rate', va='center', rotation='vertical', fontsize=16,weight='bold',fontproperties='Times New Roman')
  plt.savefig('./fig1.jpg', bbox_inches='tight', dpi=600)  
  return aucs, accuracys  
  

def figure2(files, h1, h2, et):
  fig = plt.figure(dpi=1200)
  plt.rcParams['xtick.direction'] = 'in'
  plt.rcParams['ytick.direction'] = 'in'
  plt.subplots_adjust(left=0.05,bottom=0.05,top=0.9,right=0.95,hspace=0.25,wspace=0.2)
  palette = plt.get_cmap('Set3')
  aucprs = [[] for file in files]

  for i, file in enumerate(files):
    with open(output_document + '/' + file + '_1_0_' + str(et)+ '.txt', 'rb') as f1:
      basics, train_preds, test_preds = pickle.load(f1)[:3]

    ax = plt.subplot(h1,h2,i+1)
    label = train_preds[0]
    for j, y in enumerate(train_preds[1:]):
      precision, recall = precision_recall_calculation(label, y)
      pr_auc = average_precision_score(np.array(label), np.array(y))
      print(file, method_names[j], 'auc-pr:', pr_auc)
      if i == 0:
        plt.plot(recall+[1], precision+[0], color=palette(color_indices[j]), linewidth=3, alpha=0.9, label=method_names[j])
      else:
        plt.plot(recall+[1], precision+[0], color=palette(color_indices[j]), linewidth=3, alpha=0.9)
      aucprs[i].append(pr_auc)
      
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xticks([-0.02, 0.2, 0.4, 0.6, 0.8, 1.02])
    ax.set_yticks([-0.02, 0.2, 0.4, 0.6, 0.8, 1.02])
    if i in [2,3]:
      ax.set_xticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'], weight='bold', fontsize=12,fontproperties='Times New Roman')
    else:
      ax.xaxis.set_major_formatter(plt.NullFormatter())
    if i in [0,2]:
      ax.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'], weight='bold', fontsize=12,fontproperties='Times New Roman')
    else:
      ax.yaxis.set_major_formatter(plt.NullFormatter())

    ax.text(0.95, 0.15, data_names[i], horizontalalignment='right', verticalalignment='top', transform=ax.transAxes, weight='bold', fontsize=16,fontproperties='Times New Roman')
    ax.text(-0.1, 1.025, subfigure_index[i], horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes, weight='bold', fontsize=16,fontproperties='Times New Roman')

  fig.legend(bbox_to_anchor=(0.5,-0.1), loc='upper center', frameon=False, ncol=3,prop=font1)
  fig.text(0.5, -0.05, 'Recall', ha='center',fontproperties='Times New Roman',weight='bold', fontsize=15)
  fig.text(-0.05, 0.5, 'Precision', va='center', rotation='vertical', fontsize=15,weight='bold',fontproperties='Times New Roman')
  plt.savefig('./fig2.jpg', bbox_inches='tight', dpi=600)
  return aucprs


def precision_recall_calculation(label, pred):
  scores = {}
  precision, recall = [1], [0]
  for i, x in enumerate(pred):
    if x not in scores:
      scores[x] = [0, 0]
    scores[x][1-label[i]] += 1

  all_scores = list(scores.keys())
  all_scores = sorted(all_scores, reverse=True)

  s = sum(label)
  tp, fp = 0, 0
  for i, x in enumerate(all_scores):
    value = scores[x]
    for j in range(sum(value)):
      a = value[0] / sum(value)
      tp += a
      fp += 1 - a
      precision.append(tp/(tp+fp)) 
      recall.append(tp / s)

  return precision, recall


def figure3(files, example_index, example_pro, example_method):
  fig = plt.figure(figsize=(16,12))
  plt.subplots_adjust(left=0.05,bottom=0.05,top=0.9,right=0.95,hspace=0.35,wspace=0.2)
  palette = plt.get_cmap('Set3')
  plt.rcParams['xtick.direction'] = 'in'
  plt.rcParams['ytick.direction'] = 'in'
  file = files[example_index]
  print('example-data:', file)
  print('example-method:', method_names[example_method])
  method_num = len(method_names)
  hls, acs = [[] for i in range(method_num)], [[] for i in range(method_num)]
  deltas = []
  
  for t in range(10):
    with open(output_document + '/' + file + '_'+str(example_pro)+'_'+str(t)+'_' + str(max_training_turn) + '.txt', 'rb') as f1:
      basics, train_preds, test_preds = pickle.load(f1)
    error_bound, train_num,test_num, hinge1, accuracy1 = basics
    deltas.append(error_bound)
    for i in range(1, len(test_preds)):
      if i < 3:
        ha = continuous_classifier_hinge_accuracy(train_preds[0], train_preds[i], test_preds[0], test_preds[i])
      else:
        ha = discrete_classifier_hinge_accuracy(train_preds[0], train_preds[i], test_preds[0], test_preds[i])
      hls[i-1].append(abs(ha[0] - hinge1))
      acs[i-1].append(abs(ha[1] - accuracy1))
  
  ax = plt.subplot2grid((2, 3), (0, 0), rowspan=1, colspan=1)
  # ax.tick_params(bottom=False, top=False, right=False)
  m = train_num + test_num
  upper_error = np.average(deltas) / m
  x = list(range(1, method_num))
  xx1 = [np.average(xx) for xx in hls]
  xx2 = [np.average(xx) for xx in acs]
  ax.bar(x, [xx / m for xx in xx1[:-1]], label=r'$\Delta_{test}$',color=palette(4))
  ax.bar(x, [xx / m for xx in xx2[:-1]], bottom=[xx / m for xx in xx1[:-1]], label=r'$\Delta_{train}$',color=palette(3))
  line, = ax.plot([0, method_num], [upper_error, upper_error], linewidth=4,color='gray')
  line.set_dashes((2,2))
  yl = 0.15
  ax.set_ylim([0, 1.01*yl])
  ax.set_yticks([-0.01*yl,0.05,0.1,1.01*yl])
  ax.set_xlim([0, method_num])
  ax.set_xticks(x)
  ax.set_yticklabels(['0', '0.05', '0.1', '0.15'], weight='bold', fontsize=12,fontproperties='Times New Roman')
  ax.set_xticklabels(method_names[:-1], weight='bold', fontsize=16,fontproperties='Times New Roman', rotation=30, ha='right')
    
  ax.set_ylabel(r'$\Delta$', fontsize=16, fontproperties='Times New Roman')
  ax.text(-0.1, 1.025, 'A', horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes, weight='bold', fontsize=20,fontproperties='Times New Roman')
  ax.legend(loc="upper left", fontsize=20, frameon=False)

  ax = plt.subplot2grid((2, 3), (0, 1), rowspan=1, colspan=1)
  for i in range(method_num-1):
    plt.plot([xx1[i] / (m*example_pro)], [xx2[i] / (m*(1-example_pro))], 'o', color=palette(color_indices[i]),markersize=12)
  plt.plot([0,np.average(deltas)/(m*example_pro)], [np.average(deltas)/(m*(1-example_pro)),0], linewidth=4,color='gray')
  '''
  ax_ins = inset_axes(ax, width="30%",  height="30%")
  plt.plot([xx1[method_num-2] / (m*example_pro)], [xx2[method_num-2] / (m*(1-example_pro))], 'o', color=palette(color_indices[method_num-2]),markersize=12)
  ax_ins.set_xlim([0.095, 0.105])
  ax_ins.set_ylim([0.105, 0.12])
  ax_ins.set_xticks([0.1])
  ax_ins.set_xticklabels(['0.1'], weight='bold', fontsize=12,fontproperties='Times New Roman')
  ax_ins.set_yticks([0.11,0.12])
  ax_ins.set_yticklabels(['0.11', '0.12'], weight='bold', fontsize=12,fontproperties='Times New Roman')
  '''

  xl, yl = 0.11, 0.12
  ax.set_ylim([-0.01*yl, 1.01*yl])
  ax.set_yticks([-0.01*yl,0.04,0.08, 1.01*yl])
  ax.set_xlim([-0.01*xl, 1.01*xl])
  ax.set_xticks([0.02,0.04,0.06,0.08,0.1])
  ax.set_yticklabels(['0', '0.04', '0.08','0.12'], weight='bold', fontsize=12,fontproperties='Times New Roman')
  ax.set_xticklabels(['0.02', '0.04', '0.06', '0.08','0.1'], weight='bold', fontsize=12,fontproperties='Times New Roman')
  ax.set_ylabel(r'$\Delta_{test}  $', fontsize=16, fontproperties='Times New Roman')
  ax.set_xlabel(r'$\Delta_{train}  $', fontsize=16, fontproperties='Times New Roman')
  ax.text(-0.1, 1.025, 'B', horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes, weight='bold', fontsize=20,fontproperties='Times New Roman')
  legend_elements = [Line2D([0], [0], marker='o', color=palette(color_indices[i]), label=method_names[i], markersize=12,lw=0) for i in range(method_num-1)]
  ax.text(0.1, 0.85, data_names[example_index], horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes, weight='bold', fontsize=20,fontproperties='Times New Roman')
  ax.legend(handles=legend_elements, loc="lower right", fontsize=12, frameon=False,prop=font1)
  
  ax = plt.subplot2grid((2, 3), (0, 2), rowspan=1, colspan=1)
  xx1, xx2 = [], []
  palette1 = plt.get_cmap('tab10')
  ps = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
  all_errors = []
  for p in ps:
    x1, x2 = [], []
    e = []
    for t in range(10):
      with open(output_document + '/' + file + '_'+str(example_pro)+'_'+str(t)+'_' + str(max_training_turn) + '.txt', 'rb') as f1:
        basics, train_preds, test_preds = pickle.load(f1)
      error_bound, train_num,test_num, hinge1, accuracy1 = basics
      e.append(error_bound)
      if example_method < 3:
        ha = continuous_classifier_hinge_accuracy(train_preds[0], train_preds[example_method], test_preds[0], test_preds[example_method])
      else:
        ha = discrete_classifier_hinge_accuracy(train_preds[0], train_preds[example_method], test_preds[0], test_preds[example_method])
      x1.append(abs(ha[0] - hinge1))
      x2.append(abs(ha[1] - accuracy1))
    xx1.append(np.average(x1))
    xx2.append(np.average(x2))
    all_errors.append(np.average(e))

  for i, p in enumerate(ps):
    if p in ps[-2:]:
      c = palette1(color_indices[i])
    else:
      c = palette(color_indices[i])
    plt.plot([0,e[i] / (m*p)] , [e[i] / (m*(1-p)),0], linewidth=4,color=c)
    plt.plot([xx1[i] / (m*ps[i])] , [xx2[i] / (m*(1-ps[i]))],'o', color=c, linewidth=4, alpha=0.9, markersize=12)
  
  ax_ins = inset_axes(ax, width="40%",  height="40%", loc="upper right")
  ax_ins.set_yscale('log')
  ax_ins.set_xscale('log')
  ax_ins.set_xlim([1e-3, 1e-1])
  ax_ins.set_ylim([1e-3, 1e-1])
  ax_ins.set_xticks([0.001,0.01,0.05])
  ax_ins.set_xticklabels(['0.001', '0.01', '0.05'], weight='bold', fontsize=12,fontproperties='Times New Roman')
  ax_ins.set_yticks([0.01,0.05])
  ax_ins.set_yticklabels(['0.01', '0.05'], weight='bold', fontsize=12,fontproperties='Times New Roman')
  for i, p in enumerate(ps):
    if p in ps[-2:]:
      c = palette1(color_indices[i])
    else:
      c = palette(color_indices[i])
    m1, m2 = e[i] / (m*p), e[i] / (m*(1-p))
    xs = np.linspace(1e-3,m1,1000)
    ax_ins.plot(xs, [m2-x*m2/m1 for x in xs], linewidth=4,color=c)
    # plt.plot([0,e[i] / (m*p)] , [e[i] / (m*(1-p)),0], linewidth=4,color=palette(color_indices[i]))
    # plt.plot([xx1[i] / (m*ps[i])] , [xx2[i] / (m*(1-ps[i]))],'o', color=palette(color_indices[i]), linewidth=4, alpha=0.9, markersize=12)
  
  xl, yl = 0.8, 0.35
  ax.set_ylim([-0.01*yl, yl])
  ax.set_yticks([-0.01*yl,0.1,0.2, 0.3])
  ax.set_xlim([-0.01*xl, 1.01*xl])
  ax.set_xticks([0.2,0.4,0.6,1.01*xl])
  ax.set_yticklabels(['0', '0.1', '0.2', '0.3'], weight='bold', fontsize=12,fontproperties='Times New Roman')
  ax.set_xticklabels(['0.2', '0.4', '0.6', '0.8'], weight='bold', fontsize=12,fontproperties='Times New Roman')
  ax.set_ylabel(r'$\Delta_{test} $', fontsize=16, fontproperties='Times New Roman')
  ax.set_xlabel(r'$\Delta_{train}  $', fontsize=16, fontproperties='Times New Roman')
  # ax.text(0.45, 0.85, method_names[example_method], transform=ax.transAxes, weight='bold', fontsize=20,fontproperties='Times New Roman')
  ax.text(-0.1, 1.025, 'C', horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes, weight='bold', fontsize=20,fontproperties='Times New Roman')
  legend_elements = [Line2D([0], [0], marker='o', color=palette(color_indices[i]), label=p, markersize=12,lw=4) for i, p in enumerate(ps[:-2])] + [Line2D([0], [0], marker='o', color=palette1(8), label=0.8, markersize=12,lw=4)] + [Line2D([0], [0], marker='o', color=palette1(9), label=0.9, markersize=12,lw=4)]
  
  # mark_inset(ax, ax_ins, loc1=2, loc2=2, fc="none", ec='k', lw=1)
  # mark_inset(ax, ax_ins, loc1=4, loc2=4, fc="none", ec='k', lw=1)
  ax.legend(handles=legend_elements, loc="lower right", fontsize=12, frameon=False, prop=font1)

  ax1 = plt.subplot2grid((2, 3), (1, 0), rowspan=1, colspan=1)
  results = []
  for file in all_file:
    with open('./bounds_'+file, 'rb') as f1:
      results.append(pickle.load(f1))
  markers = ['o', '^', 's', '+', 'x', 'd', '*']
  ins_set = [0, 1]
  for i in range(6):
    if i in ins_set:
      continue
    result = results[i][1]
    plt.plot(ps+[1], [x[0] for x in result[1:]],markers[i]+'-', color=palette(color_indices[i]), linewidth=4, alpha=0.9,label=data_names[i],markersize=12)
    
  ax_ins = inset_axes(ax1,width="30%",  height="30%", loc="upper left")
  for i in ins_set:
    result = results[i][1]
    plt.plot(ps+[1], [x[0] for x in result[1:]],markers[i]+'-', color=palette(color_indices[i]), linewidth=2, alpha=0.9,label=data_names[i],markersize=6)
  ax_ins.yaxis.tick_right()
  ax_ins.set_xlim([0, 1.1])
  ax_ins.set_ylim([0.23, 0.37])
  ax_ins.set_xticks([0.1,0.5,0.9])
  ax_ins.set_xticklabels(['0.1', '0.5', '0.9'], weight='bold', fontsize=12,fontproperties='Times New Roman')
  ax_ins.set_yticks([0.25,0.3,0.35])
  ax_ins.set_yticklabels(['0.25', '0.3', '0.35'], weight='bold', fontsize=12,fontproperties='Times New Roman')

  ax1.set_xlim([0.09, 1.01])
  ax1.set_xticks([0.09]+[0.1*j for j in range(2, 10)]+[1.01])
  ax1.set_xticklabels(['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9','1'], weight='bold', fontsize=12,fontproperties='Times New Roman')
  ax1.set_ylim([0, 0.1])
  ax1.set_yticks([0.02,0.04,0.06,0.08,0.1])
  ax1.set_yticklabels(['0.02', '0.04', '0.06', '0.08', '0.1'], weight='bold', fontsize=12,fontproperties='Times New Roman')
  ax1.text(-0.1, 1.025, 'D', horizontalalignment='left', verticalalignment='bottom', transform=ax1.transAxes, weight='bold', fontsize=20,fontproperties='Times New Roman')
  # sub-titles
  ax1.set_xlabel(r'$|\mathcal{S}_{train}|/|\mathcal{S}|$', fontsize=16, fontproperties='Times New Roman')
  ax1.set_ylabel('minimum hinge loss', fontsize=16, fontproperties='Times New Roman')
  legend_elements = [Line2D([0], [0], marker=markers[i], color=palette(color_indices[i]), label=data_names[i], markersize=12,lw=4) for i in range(6)]
  # ax1.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(0.05,0.95), fontsize=12, frameon=False, prop=font1, ncol=2)

  ax2 = plt.subplot2grid((2, 3), (1, 1), rowspan=1, colspan=1)
  for i in range(6):
    if i in ins_set:
      continue
    result = results[i][0]
    plt.plot([0]+ps, [min(x[1],1) for x in result[:-1]],markers[i]+'-', color=palette(color_indices[i]), linewidth=4, alpha=0.9, markersize=12)

  ax_ins = inset_axes(ax2, loc='lower right', width="30%",  height="30%")
  for i in ins_set:
    result = results[i][0]
    plt.plot([0]+ps, [min(x[1],1) for x in result[:-1]],markers[i]+'-', color=palette(color_indices[i]), linewidth=4, alpha=0.9, markersize=6)

  ax_ins.xaxis.tick_top()
  ax_ins.set_xlim([-0.05, 0.95])
  ax_ins.set_ylim([0.635,0.76])
  ax_ins.set_xticks([0.1,0.5,0.9])
  ax_ins.set_xticklabels(['0.1', '0.5', '0.9'], weight='bold', fontsize=12,fontproperties='Times New Roman')
  ax_ins.set_yticks([0.65,0.7, 0.75])
  ax_ins.set_yticklabels(['0.65', '0.7', '0.75'], weight='bold', fontsize=12,fontproperties='Times New Roman')

  ax2.set_xlim([-0.01, 0.91])
  ax2.set_xticks([-0.01]+[0.1*j for j in range(1, 9)]+[0.91])
  ax2.set_xticklabels(['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9'], weight='bold', fontsize=12,fontproperties='Times New Roman')
  ax2.set_ylim([0.88, 1.01])
  ax2.set_yticks([0.88,0.92,0.96,1.01])
  ax2.set_yticklabels(['0.88', '0.92', '0.96', '1'], weight='bold', fontsize=12,fontproperties='Times New Roman')
  ax2.text(-0.1, 1.025, 'E', horizontalalignment='left', verticalalignment='bottom', transform=ax2.transAxes, weight='bold', fontsize=20,fontproperties='Times New Roman')
  # sub-titles
  ax2.set_xlabel(r'$|\mathcal{S}_{train}| / |\mathcal{S}|$', fontsize=16, fontproperties='Times New Roman')
  ax2.set_ylabel(r'$\mathrm{AC}^u$', fontsize=16, fontproperties='Times New Roman')
  ax2.legend(handles=legend_elements, loc="lower left", bbox_to_anchor=(0.05,0.05), fontsize=12, frameon=False, prop=font1, ncol=2)

  ax3 = plt.subplot2grid((2, 3), (1, 2), rowspan=1, colspan=1)
  for i in range(1, 6):
    result = results[i]
    result0 = result[1]
    plt.plot(ps, [x[2] for x in result0[1:-1]],markers[i]+'-', color=palette(color_indices[i]), linewidth=4, alpha=0.9, markersize=12)

  ax3.set_xlim([0.09, 0.91])
  ax3.set_xticks([0.1*j for j in range(1, 10)])
  ax3.set_xticklabels(['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9'], weight='bold', fontsize=12,fontproperties='Times New Roman')
  ax3.set_ylim([0, 0.012])
  ax3.set_yticks([0,0.004,0.008,0.012])
  ax3.set_yticklabels(['0', '0.004', '0.008', '0.012'], weight='bold', fontsize=12,fontproperties='Times New Roman')
  ax3.text(-0.1, 1.025, 'F', horizontalalignment='left', verticalalignment='bottom', transform=ax3.transAxes, weight='bold', fontsize=20,fontproperties='Times New Roman')
  # sub-titles
  ax3.set_xlabel(r'$|\mathcal{S}_{train}|/|\mathcal{S}|$', fontsize=16, fontproperties='Times New Roman')
  ax3.set_ylabel(r'$\Delta$', fontsize=16, fontproperties='Times New Roman')
  # ax3.legend(handles=legend_elements, loc="lower center", bbox_to_anchor=(0.5,0.05), fontsize=12, frameon=False, prop=font1, ncol=2)

  plt.savefig('./fig3.jpg', dpi=600)


def bound_calculation(file):
  dataset = Data('./cleaned_dataset/' + file, file, 0.5)
  bound = dataset.theoretical_error()
  with open('./bounds_'+file, 'wb') as f:
    pickle.dump(bound, f)


def uc_bound_calculation(file):
  dataset = Data('./cleaned_UC_dataset/' + file, file, 0.5)
  bound = dataset.theoretical_error()
  with open('./bounds_'+file, 'wb') as f:
    pickle.dump(bound, f)


def table1(ar, ap, ac):
  with open('./table1.txt', 'w') as f:
    for i in range(len(ar)):
      auc_roc = ar[i]
      auc_pr = ap[i]
      acc = ac[i]
      f.write('&'.join(['\\multirow{3}*{' + data_names[i] + '}', 'AR'] + [str(round(x, 4)) for x in auc_roc ])+'\\\\\n')
      f.write('&'.join(['~', 'AP'] + [str(round(x, 4)) for x in auc_pr ])+'\\\\\n')
      f.write('&'.join(['~', 'AC'] + [str(round(x, 4)) for x in acc ])+'\\\\\n\\hline\n')


def figureS1(j1, ys):
  fig = plt.figure(figsize=(16,16))
  plt.subplots_adjust(left=0.05,bottom=0.05,top=0.9,right=0.95,hspace=0.35,wspace=0.2)
  palette = plt.get_cmap('Set3')
  plt.rcParams['xtick.direction'] = 'in'
  plt.rcParams['ytick.direction'] = 'in'
  all_pros = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
  method_num = len(method_names)
  labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
  file = all_file[j1]
  
  for i1, example_pro in enumerate(all_pros):

    hls, acs = [[] for i in range(method_num)], [[] for i in range(method_num)]
    deltas = []
    for t in range(10):
      try:
        with open(output_document + '/' + file + '_'+str(example_pro)+'_'+str(t)+'_' + str(max_training_turn) + '.txt', 'rb') as f1:
          basics, train_preds, test_preds = pickle.load(f1)
      except:
        continue
      error_bound, train_num,test_num, hinge1, accuracy1 = basics
      deltas.append(error_bound)
      for i in range(1, len(test_preds)):
        if i < 3:
          ha = continuous_classifier_hinge_accuracy(train_preds[0], train_preds[i], test_preds[0], test_preds[i])
        else:
          ha = discrete_classifier_hinge_accuracy(train_preds[0], train_preds[i], test_preds[0], test_preds[i])
        hls[i-1].append(abs(ha[0] - hinge1))
        acs[i-1].append(abs(ha[1] - accuracy1))
    
    c1, c2 = i1 // 3, i1 % 3 
    ax = plt.subplot2grid((3, 3), (c1, c2), rowspan=1, colspan=1)
    # ax.tick_params(bottom=False, top=False, right=False)
    m = train_num + test_num
    upper_error = np.average(deltas) / m
    x = list(range(1, method_num))
    xx1 = [np.average(xx) for xx in hls]
    xx2 = [np.average(xx) for xx in acs]
    ax.bar(x, [xx / m for xx in xx1[:-1]], label=r'$\Delta_{\text{test}}$',color=palette(4))
    ax.bar(x, [xx / m for xx in xx2[:-1]], bottom=[xx / m for xx in xx1[:-1]], label=r'$\Delta_{\text{train}}$',color=palette(3))
    line, = ax.plot([0, method_num], [upper_error, upper_error], linewidth=4,color='gray')
    line.set_dashes((2,2))

    ax.set_ylim([0, ys[-1]])
    ax.set_yticks(ys)
    ax.set_xlim([0, method_num])
    ax.set_xticks(x)
    
    # ax.set_ylabel(r'$\Delta$', fontsize=16, fontproperties='Times New Roman')
    ax.text(-0.1, 1.025, labels[i1], horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes, fontsize=20, weight='bold',fontproperties='Times New Roman')
    if i1 == 0:
      ax.legend(loc="upper left", fontsize=20, frameon=False, weight='bold',fontproperties='Times New Roman')
    ax.text(0.95, 0.95, 'p='+str(example_pro), horizontalalignment='right', verticalalignment='top', transform=ax.transAxes, weight='bold', fontsize=16,fontproperties='Times New Roman')

    if c2 == 0:
      ax.set_yticklabels([str(yy) for yy in ys], weight='bold', fontsize=12,fontproperties='Times New Roman')
    else:
      ax.yaxis.set_major_formatter(plt.NullFormatter())
    if c1 == 2:
      ax.set_xticklabels(method_names[:-1], weight='bold', fontsize=16,fontproperties='Times New Roman', rotation=30, ha='right')
    else:
      ax.xaxis.set_major_formatter(plt.NullFormatter())

  fig.text(-0.02, 0.5, r'$\Delta$', va='center', rotation='vertical', fontsize=40,weight='bold',fontproperties='Times New Roman')
  plt.savefig('./figS' + str(j1 + 1) + '.jpg', bbox_inches='tight', dpi=600)


if __name__ == '__main__':
  # parameters setting
  method_names = ['XGBoost+square', 'XGBoost+logistic', 'XGBoost+hinge', 'XGBoost+softmax', 'MLP', 'optimal']
  color_indices = [0,2,4,5,6,9,3,8,10]
  font1 = {'family' : 'Times New Roman', 'weight' : 'bold', 'size': 12}
  output_document = './output'
  error_output = './bounds.txt'
  max_training_turn = 10
  all_file = ['airlines.csv', 'heart.csv', 'income_evaluation.csv', 'SleepStudyData.csv']
  data_names = ['AID', 'HED', 'INE', 'STS']
  subfigure_index = ['A', 'B', 'C', 'D']
  division_radio = 0.7
  print(all_file)
  ### figure 1, figure 2 and table 1: training set / dataset = 1
  # experiment
  # experiment(all_file[1], 1, 0, max_training_turn)
  # for file in all_file:
  #   experiment(file, 1, 0, max_training_turn)
  # fs, ppp, ixs, ems = [], [], [], []
  # for file in all_file:
  #   for i in range(10):
  #     for p in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
  #       fs.append(file)
  #       ppp.append(p)
  #       ixs.append(i)
  #       ems.append(max_training_turn)
  # parallel(experiment, fs, ppp, ixs, ems)
  
  # results visualization
  # col_num = 2
  # row_num = 2
  # ar, ac = figure1(all_file, row_num, col_num, max_training_turn)
  # ap = figure2(all_file, row_num, col_num, max_training_turn)
  # table1(ar, ap, ac)
  
  ### figure 3: training set / dataset = 0.7
  # experiment
  for file in all_file:
    bound_calculation(file)

  # results visualization
  # figure3(all_file, example_index=2, example_pro=0.7, example_method=2)

  # figureS1(0, [0, 0.02, 0.04, 0.06, 0.08])
  # figureS1(1, [0, 0.01, 0.02, 0.03, 0.04, 0.05])
  # figureS1(2, [0, 0.05, 0.1, 0.15])
  # figureS1(3, [0, 0.002, 0.004, 0.006, 0.008, 0.01])
  # figureS1(4, [0, 0.1, 0.2, 0.3])
  # figureS1(5, [0, 0.1, 0.2, 0.3, 0.4, 0.5])

  # for file in list(os.listdir('./cleaned_UC_dataset')):
  #   uc_experiment(file, 1, 0, max_training_turn)