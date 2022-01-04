import subprocess
import wandb
import yaml
import numpy as np

sweep_config = {
    'name': 'Scaling Laws for Neural Language Models sweep',
    'method': 'grid',
    'metric': {
        'name': 'loss',
        'goal': 'minimize'
    },
    'parameters': {
        # top-level params can be passed as they are
        'warmup': {
            'values': [
                       # multiple values will be swept over
                       # in grid fashion
                       0.01
                       ]
        },
        # params that are constant can be changed in template
        # or set to single values
        'steps_per_print': {
            'values': [
                       10
            ]
        },
        # this is a placeholder for calculated values below
        'valid_set': {
          'values': [
                    # This will be a list of lists
                     # one for each valid scaling law
                     # sweep configuration
          ]
        },
        'opt_configs': {
            'values': [
                       #Placeholder for list of lists,
                       #one for each valid optimizer
                       #sweep configuration
            ]
        }
    }
}

param_dict = {
  # these are the ranges to sweep over
  # during the scaling laws sweep
  # ballpark numbers from Figure 5
  'exponent': [exponent for exponent in range(10,22)],
  'ar': [round(10**x) for x in np.linspace(1,2.5,3)],
  'attn_dim': [round(10**x) for x in np.linspace(1.5,2.5,3)],
  'opts': {
      #These are optimizer-specific configs
      #Each optimizer can take different args
      #If type of optimizer does not match
      #template yaml, must specify all args
      #except LR (which is calculated in scaling laws)
      'adam': {
          'betas': [
                    # list of lists expected
                    # (as betas takes two values)
                    [0.9, 0.999],
                    [0.95, 0.95]
                    ],
          'eps': [1e-7]
      },
      'sm3': {
          'momentum': [0.0, 0.01],
          'beta': [0.0, 0.01],
          'eps': [1e-30]
      }

  }
}

def sweep_to_yml(run_config,template_yml,dummy_yml):
  """
  Write the current run config to a dummy yaml
  """
  with open(template_yml, 'r') as f:
    conf = yaml.load(f,Loader=yaml.Loader)
  for k in run_config.keys():
    if k == 'optimizer':
      # need to overwrite template config if optimizer differs
      if run_config['optimizer']['type'] != conf['optimizer']['type']:
        conf['optimizer'] = run_config['optimizer']
      else:
        for c in run_config['optimizer']['params'].keys():
          conf['optimizer']['params'][c] = run_config['optimizer']['params'][c]
    else:
      conf[k] = run_config[k]
  with open(dummy_yml, 'w') as f:
    yaml.dump(conf, f)

def grid(input,depth):
  """
  Find all combinations of the input elements
  Depth is for recursion
  """
  if depth == len(input)-1:
    return input[depth]
  else:
    output = []
    for i in input[depth]:
      for j in grid(input,depth+1):
        if type(i) == list:
          output.append([i,j])
        else:
          if type(j) == list:
            output.append([i]+j)
          else:
            output.append([i,j])
    return output

def make_row(input,label):
  """
  Call grid to make a valid combination
  of hparams
  """
  d = input.copy()
  raw = grid([*d.values()],0)
  output = []
  for r in raw:
    row = [label,r,list(d.keys())]
    output.append(row)
  return output

for exponent in param_dict['exponent']:
  N = np.exp(exponent)
  # add LR according equation D.1 from Kaplan et. al
  # "Scaling Laws for Neural Language Models"
  lr = 0.003239 + (-0.0001395)*np.log(N)
  for ar in param_dict['ar']:
    # substitute for n_layer, solve for d_model
    d_model = (N*ar/12)**(1/3)
    # calculate n_layer
    n_layer = N/12/(d_model**2)
    if n_layer < 1:
      # don't clip n_layer
      break
    for attn_dim in param_dict['attn_dim']:
      # add n_head per attn_dim
      n_head = d_model/attn_dim
      if n_head < 1:
        # don't clip n_head
        break
      # change d_model to nearest multiple of n_layer
      n_layer = round(n_layer)
      d_model = n_layer*round(d_model/n_layer)
      # add this combination as a string to sweep_config
      sweep_config['parameters']['valid_set']['values'].append(
              [round(x) for x in [n_layer, d_model, n_head]] + \
              [float(lr)]
      )

for opt_type in param_dict['opts'].keys():
  for row in make_row(param_dict['opts'][opt_type],opt_type):
    sweep_config['parameters']['opt_configs']['values'].append(row)

# test run
def train():
  run = wandb.init()
  # convert wandb Config class to dictionary
  config_dict = run.config.as_dict()
  valid_set = {k:v for k,v in zip(
        # these are from neox_arguments.md
        [
         'num-layers', # "n_layers" (GPT)
         'hidden-size', # "d_model" (GPT)
         'num-attention-heads', # "n_heads" (GPT)
         'lr' # "learning_rate" (GPT)
         ],
        [float(x) for x in config_dict.pop('valid_set')]
    )}
  # need to convert floats to int
  for key in ['num-layers', 'hidden-size', 'num-attention-heads']:
      valid_set[key] = int(valid_set[key])
  # initialize optimizer params
  config_dict['optimizer'] = {
      'params': {
          # optimizer parameters will go here
      }
  }
  # fill in optimizer params
  for k in config_dict.keys():
    if k == 'opt_configs':
      config_dict['optimizer']['type'] = config_dict['opt_configs'][0]
      opt_dict = {
          k:v for k,v in zip(
              config_dict['opt_configs'][-1],
              config_dict['opt_configs'][1])
          }
      for c,v in opt_dict.items():
        config_dict['optimizer']['params'][c] = v
  # remove optimizer params from top-level
  [config_dict.pop(k) for k in \
   [c for c in config_dict.keys() if c[:4]=='opt_']
   ]
  # set learning rate in optimizer parameters
  config_dict['optimizer']['params']['lr'] = valid_set.pop('lr')
  # transfer top-level params to run.config
  for k in valid_set.keys():
    config_dict[k] = valid_set[k]
  # print the run config
  print(config_dict)
  # write the run config to the dummy yaml
  sweep_to_yml(
      config_dict,
      template_yml='configs/small.yml',
      dummy_yml='configs/dummy.yml'
      )
  # execute the actual pretrain process
  cmd = subprocess.run('python deepy.py pretrain_gpt2.py -d configs dummy.yml local_setup.yml'.split(' '))
  run.finish()

sweep_id = wandb.sweep(sweep_config)
agent = wandb.agent(sweep_id=sweep_id, function=train)
agent.run()
