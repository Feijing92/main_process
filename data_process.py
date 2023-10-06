import os
import json


def json_read(json_document):
  file_names = list(os.listdir(json_document))
  user_datas = []
  tweet_datas = []
  for name in file_names:
    with open(json_document + '/' + name, 'rb') as f:
      json_data = json.load(f)
    if name[:4] == 'user':
      user_datas.append(json_data)
    elif name[:4] == 'data':
      tweet_datas.append(json_data)
    # print(json_data)
  return user_datas, tweet_datas


if __name__ == '__main__':
  document = './data-leader/2019 India Climate Strikes Movement'
  users, tweets = json_read(document)
