import os
import sys
import pickle
import json
import shutil

def write_interaction(interaction_list,split,output_dir):
  json_split = os.path.join(output_dir,split+'.json')
  pkl_split = os.path.join(output_dir,split+'.pkl')

  with open(json_split, 'w') as outfile:
    for interaction in interaction_list:
      json.dump(interaction, outfile, indent = 4)
      outfile.write('\n')

  new_objs = []
  for i, obj in enumerate(interaction_list):
    new_interaction = []
    for ut in obj["interaction"]:
      sql = ut["sql"]
      sqls = [sql]
      tok_sql_list = []
      for sql in sqls:
        results = []
        tokenized_sql = sql.split()
        tok_sql_list.append((tokenized_sql, results))
      ut["sql"] = tok_sql_list
      new_interaction.append(ut)
    obj["interaction"] = new_interaction
    new_objs.append(obj)

  with open(pkl_split,'wb') as outfile:
    pickle.dump(new_objs, outfile)

  return


def read_database_schema(database_schema_filename, schema_tokens, column_names, database_schemas_dict):
  with open(database_schema_filename) as f:
    database_schemas = json.load(f)

  def get_schema_tokens(table_schema):
    column_names_surface_form = []
    column_names = []
    # column_names = table_schema['column_names']
    column_names_original = table_schema['column_names_original']
    table_names = table_schema['table_names']
    table_names_original = table_schema['table_names_original']
    for i, (table_id, column_name) in enumerate(column_names_original):
      if table_id >= 0:
        table_name = table_names_original[table_id]
        column_name_surface_form = '{}.{}'.format(table_name,column_name)
      else:
        # this is just *
        column_name_surface_form = column_name
      column_names_surface_form.append(column_name_surface_form.lower())
      column_names.append(column_name.lower())

    # also add table_name.*
    for table_name in table_names_original:
      column_names_surface_form.append('{}.*'.format(table_name.lower()))

    return column_names_surface_form, column_names

  for table_schema in database_schemas:
    database_id = table_schema['db_id']
    database_schemas_dict[database_id] = table_schema
    schema_tokens[database_id], column_names[database_id] = get_schema_tokens(table_schema)

  return schema_tokens, column_names, database_schemas_dict


def read_sparc_split(split_json, interaction_list):
  with open(split_json) as f:
    split_data = json.load(f)
  print('read_sparc_split', split_json, len(split_data))

  for interaction_data in split_data:
    db_id = interaction_data['database_id']
    final_sql = interaction_data['final']['query']
    final_utterance = interaction_data['final']['utterance']

    if db_id not in interaction_list:
      interaction_list[db_id] = []

    # no interaction_id in train
    if 'interaction_id' in interaction_data['interaction']:
      interaction_id = interaction_data['interaction']['interaction_id']
    else:
      interaction_id = len(interaction_list[db_id])

    interaction = {}
    interaction['id'] = ''
    interaction['scenario'] = ''
    interaction['database_id'] = db_id
    interaction['interaction_id'] = interaction_id
    interaction['final'] = {}
    interaction['final']['utterance'] = final_utterance
    interaction['final']['sql'] = final_sql
    interaction['interaction'] = []

    for turn in interaction_data['interaction']:
      turn_sql = []
      print_final = False
      for query_tok in turn['query_toks_no_value']:
        if query_tok != '.' and '.' in query_tok:
          # invalid sql; didn't use table alias in join
          turn_sql += query_tok.replace('.',' . ').split()
          print_final = True
        else:
          turn_sql.append(query_tok)
      turn_sql = ' '.join(turn_sql)

      # Correct some human sql annotation error
      turn_sql = turn_sql.replace('select f_id from files as t1 join song as t2 on t1 . f_id = t2 . f_id', 'select t1 . f_id from files as t1 join song as t2 on t1 . f_id = t2 . f_id')
      turn_sql = turn_sql.replace('select name from climber mountain', 'select name from climber')
      turn_sql = turn_sql.replace('select sid from sailors as t1 join reserves as t2 on t1 . sid = t2 . sid join boats as t3 on t3 . bid = t2 . bid', 'select t1 . sid from sailors as t1 join reserves as t2 on t1 . sid = t2 . sid join boats as t3 on t3 . bid = t2 . bid')
      turn_sql = turn_sql.replace('select avg ( price ) from goods )', 'select avg ( price ) from goods')
      turn_sql = turn_sql.replace('select min ( annual_fuel_cost ) , from vehicles', 'select min ( annual_fuel_cost ) from vehicles')
      turn_sql = turn_sql.replace('select * from goods where price < ( select avg ( price ) from goods', 'select * from goods where price < ( select avg ( price ) from goods )')
      turn_sql = turn_sql.replace('select distinct id , price from goods where price < ( select avg ( price ) from goods', 'select distinct id , price from goods where price < ( select avg ( price ) from goods )')
      turn_sql = turn_sql.replace('select id from goods where price > ( select avg ( price ) from goods', 'select id from goods where price > ( select avg ( price ) from goods )')

      if print_final and 'train' in split_json:
        continue

      if 'utterance_toks' in turn:
        turn_utterance = ' '.join(turn['utterance_toks']) # not lower()
      else:
        turn_utterance = turn['utterance']

      interaction['interaction'].append({'utterance': turn_utterance, 'sql': turn_sql})
    interaction_list[db_id].append(interaction)

  return interaction_list


def read_sparc(sparc_dir, interaction_list):
  train_json = os.path.join(sparc_dir, 'train_no_value.json')
  interaction_list = read_sparc_split(train_json, interaction_list)

  dev_json = os.path.join(sparc_dir, 'dev_no_value.json')
  interaction_list = read_sparc_split(dev_json, interaction_list)

  return interaction_list


def preprocess():
  # Validate output_vocab
  output_vocab = ['_UNK', '_EOS', '.', 't1', 't2', '=', 'select', 'from', 'as', 'value', 'join', 'on', ')', '(', 'where', 't3', 'by', ',', 'count', 'group', 'order', 'distinct', 't4', 'and', 'limit', 'desc', '>', 'avg', 'having', 'max', 'in', '<', 'sum', 't5', 'intersect', 'not', 'min', 'except', 'or', 'asc', 'like', '!', 'union', 'between', 't6', '-', 't7', '+', '/']
  print('size of output_vocab', len(output_vocab))
  print('output_vocab', output_vocab)
  print()

  sparc_dir = 'data/sparc/'
  database_schema_filename = 'data/sparc/tables.json'

  output_dir = 'data/sparc_data'
  if os.path.isdir(output_dir):
    shutil.rmtree(output_dir)
  os.mkdir(output_dir)

  schema_tokens = {}
  column_names = {}
  database_schemas = {}

  print('Reading spider database schema file')
  schema_tokens, column_names, database_schemas = read_database_schema(database_schema_filename, schema_tokens, column_names, database_schemas)
  num_database = len(schema_tokens)
  print('total number of schema_tokens / databases:', len(schema_tokens))

  output_database_schema_filename = os.path.join(output_dir, 'tables.json')
  with open(output_database_schema_filename, 'w') as outfile:
    json.dump([v for k,v in database_schemas.items()], outfile, indent=4)

  def read_split(data_dir):
    train_database = []
    with open(os.path.join(data_dir,'train_db_ids.txt')) as f:
      for line in f:
        train_database.append(line.strip())

    dev_database = []
    with open(os.path.join(data_dir,'dev_db_ids.txt')) as f:
      for line in f:
        dev_database.append(line.strip())

    return train_database, dev_database


  train_database, dev_database = read_split(sparc_dir)

  interaction_list = {}

  interaction_list = read_sparc(sparc_dir, interaction_list)
  print('interaction_list length', len(interaction_list))

  print('num_database', num_database, len(train_database), len(dev_database))

  train_interaction = []
  for database_id in interaction_list:
    # Include spider train
    if database_id not in dev_database:
      train_interaction += interaction_list[database_id]

  dev_interaction = []
  for database_id in dev_database:
    dev_interaction += interaction_list[database_id]

  print('train interaction: ', len(train_interaction))
  print('dev interaction: ', len(dev_interaction))

  write_interaction(train_interaction, 'train', output_dir)
  write_interaction(dev_interaction, 'dev', output_dir)

  return


if __name__ == '__main__':
  preprocess()


