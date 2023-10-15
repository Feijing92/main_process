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
  G = nx.DiGraph()
  for link in new_links:
    G.add_edge(link[0], link[1])
  print(G.number_of_nodes(), G.number_of_edges())

  components = sorted(nx.weakly_connected_components(G), reverse=True, key=lambda x: len(x))
  print(len(components), [len(x) for x in components])

  g1 = G.subgraph(components[0])
  g2 = G.subgraph(components[1])

  subgraph_analysis(g1, 'g1')
  subgraph_analysis(g2, 'g2')
  

def list_statistics(l):
  lt = {}
  for x in l:
    if x not in lt.keys():
      lt[x] = 1
    else:
      lt[x] += 1
  return lt


def subgraph_analysis(g, name):
  print('*' * 40)
  print(nx.is_weakly_connected(g))
  print(name + ':', g.number_of_nodes(), g.number_of_edges())
  strongly_components = sorted(nx.strongly_connected_components(g), reverse=True, key=lambda x: len(x))
  print('# of strongly_components:', len(strongly_components))
  print('frequency distritbuion of strongly_components:', list_statistics([len(xx) for xx in strongly_components]))
  nx.write_gexf(g, name + '.gexf')

  core_nodes = strongly_components[0]
  g_core = g.subgraph(core_nodes)
  core_edges = g_core.edges()
  print('cores:', len(core_nodes), len(core_edges))
  node2sc = {}
  component2layer = {}
  for sc_index, sc in enumerate(strongly_components):
    component2layer[sc_index] = 0
    for node in sc:
      node2sc[node] = sc_index
  
  for a, b in g.edges():
    ca, cb = node2sc[a], node2sc[b]
    if ca != cb:
      component2layer[cb] = -1
  
  layer_num = 0
  while True:
    # print(component2layer)
    for a, b in g.edges():
      ca, cb = node2sc[a], node2sc[b]
      if ca == cb:
        continue
      if component2layer[ca] == layer_num:
        if component2layer[cb] == -1:
          component2layer[cb] = layer_num + 1
    
    if min(list(component2layer.values())) == 0:
      break
    else:
      layer_num += 1
  
  layers = max(list(component2layer.values())) + 1
  layer_nodes = [[] for x in range(layers)]
  for key, value in component2layer.items():
    for node in strongly_components[key]:
      layer_nodes[value].append(node)
  
  print([len(x) for x in layer_nodes], component2layer[0])
  print(list_statistics([(component2layer[node2sc[a]], component2layer[node2sc[b]]) for a, b in g.edges()]))

  communities = sorted(nx.algorithms.community.greedy_modularity_communities(g), reverse=True, key=lambda x: len(x))
  print(list_statistics([len(x) for x in communities]))
  print(len(communities), nx.algorithms.community.modularity(g, communities))


if __name__ == '__main__':
  document = './data-leader/2019 India Climate Strikes Movement'
  users, tweets = json_read(document)
  reply1, mention1, reference1 = user_data_processing(users)
  reply2, mention2, reference2 = tweet_data_processing(tweets)
  simple_graph(reply1 + mention1 + reference1 + reply2 + mention2 + reference2)
