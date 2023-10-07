import os
import json
import networkx as nx
import matplotlib as plt


def json_print(json_string):
  s1 = json.dumps(json_string, sort_keys=True, indent=4, separators=(', ', ': '))
  print(s1)


def json_read(json_document):
  file_names = list(os.listdir(json_document))
  user_datas = []
  tweet_datas = []
  for name in file_names:
    
    if name[:4] == 'user':
      with open(json_document + '/' + name, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
      user_datas.append(json_data)
      
    elif name[:4] == 'data':
      with open(json_document + '/' + name, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
      tweet_datas.append(json_data)
    
  return user_datas, tweet_datas


def user_data_processing(datas):
  groups = []
  mention_relationships = []  # (user, user, tweet)
  reply_relationships = []  # (user, user, tweet)
  reference_relationships = []  # (user, user, tweet)
  tweets_history = {}

  for data in datas:
    group = []
    for x in data['users']:
      group.append(x['id'])
    groups.append(group)

    tweets = data['tweets']
    mentions, replys, references, history = tweet_reading(tweets)
    mention_relationships += mentions
    reply_relationships += replys
    reference_relationships += references
    for key, value in history.items():
      if key not in tweets_history:
        tweets_history[key] = value
      else:
        tweets_history[key] += value

  return history_check(reply_relationships, mention_relationships, reference_relationships, tweets_history)


def tweet_data_processing(datas):
  mention_relationships = []  # (user, user, tweet)
  reply_relationships = []  # (user, user, tweet)
  reference_relationships = []  # (user, user, tweet)
  tweets_history = {}

  for data in datas:
    mentions, replys, references, history = tweet_reading(data)
    mention_relationships += mentions
    reply_relationships += replys
    reference_relationships += references
    for key, value in history.items():
      if key not in tweets_history:
        tweets_history[key] = value
      else:
        tweets_history[key] += value

  return history_check(reply_relationships, mention_relationships, reference_relationships, tweets_history)


def history_check(reply, mention, reference, history):
  reply = list(set(reply))
  mention = list(set(mention))
  reference = list(set(reference))

  def check(l, h):
    removed = []
    for x in l:
      a, b, c = x
      if c in h:
        d = h[c]
      if (a, b, d) in l:
        removed.append((a, b, d))
    for x in removed:
      l.remove(x)
  
  check(reply, history)
  check(mention, history)
  check(reference, history)

  print(len(reply), len(mention), len(reference))

  return reply, mention, reference


def tweet_reading(tweet_datas):
  mention_relationships = []  # (user, user, tweet)
  reply_relationships = []  # (user, user, tweet)
  reference_relationships = []  # (user, user, tweet)
  tweets_history = {}

  for tweet in tweet_datas:
    author_id = tweet["author_id"]
    conversation_id = tweet["conversation_id"]
    if "edit_history_tweet_ids" in tweet.keys():
      if len(tweet["edit_history_tweet_ids"]) > 0:
        tweets_history[conversation_id] = tweet["edit_history_tweet_ids"]
    
    mentions = tweet["entities"]["mentions"]
    for mention in mentions:
      mention_relationships.append((author_id, mention["id"], conversation_id))
    
    if "in_reply_to_user_id" in tweet.keys():
      reply_id = tweet["in_reply_to_user_id"]
      reply_relationships.append((author_id, reply_id, conversation_id))
    
    if "referenced_tweets" in tweet.keys():
      for x in tweet["referenced_tweets"]:
        reference_relationships.append((author_id, x["id"], conversation_id))

  return mention_relationships, reply_relationships, reference_relationships, tweets_history


def simple_graph(links):
  new_links = list(set(links))
  G = nx.Graph()
  for link in new_links:
    G.add_edge(link[0], link[1])
  print(G.number_of_nodes(), G.number_of_edges())

  components = sorted(nx.connected_components(G), reverse=True, key=lambda x: len(x))
  print(len(components), [len(x) for x in components])

  subgraph_analysis(G.subgraph(components[0]), 'g1')
  subgraph_analysis(G.subgraph(components[1]), 'g2')
  

def subgraph_analysis(g, name):
  print('*' * 40)
  print(name + ':')
  print(g.number_of_nodes(), g.number_of_edges())
  nx.write_gexf(g, name + '.gexf')

  communities = sorted(nx.algorithms.community.greedy_modularity_communities(g), reverse=True, key=lambda x: len(x))
  print([len(x) for x in communities])
  print(len(communities), nx.algorithms.community.modularity(g, communities))


if __name__ == '__main__':
  document = './data-leader/2019 India Climate Strikes Movement'
  users, tweets = json_read(document)
  reply1, mention1, reference1 = user_data_processing(users)
  reply2, mention2, reference2 = tweet_data_processing(tweets)
  simple_graph(reply1 + mention1 + reference1 + reply2 + mention2 + reference2)
