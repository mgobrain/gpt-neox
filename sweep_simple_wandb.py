import wandb
import numpy as np
import yaml
import glob
import subprocess

# change this to argparse in the future
template_yml = 'configs/smaller.yml'
dummy_yml = 'configs/dummy.yml'
sweep_yml = 'configs/sweep_config.yml'
project = "neox_sweep"

# read template to dictionary
with open(template_yml, 'r') as f:
    base_conf = yaml.safe_load(f)

# convert nested dictionaries to strings
new_conf = {}
old_keys = []
for k in base_conf:
    if type(base_conf[k]) == dict:
        old_keys.append(k)
        for k2 in base_conf[k]:
            new_conf[k+'.'+k2] = base_conf[k][k2]

base_conf.update(new_conf)

# remove redundant settings
for k in set(old_keys):
    base_conf.pop(k, None)

sweep_config = {
    'name': 'neox_sweep',
    'method': 'grid',
    'metric': {
        'name': 'lm_loss',
        'goal': 'minimize'
    }
}

# initialize sweep_config with template
sweep_config['parameters'] = {
    k: {'values': [v]} for k, v in zip(
        base_conf.keys(), base_conf.values()
    )
}

# update sweep_config with user-defined sweep
if sweep_yml:
    with open(sweep_yml, 'r') as f:
        user_conf = yaml.safe_load(f)
        sweep_config['parameters'].update(user_conf)

def train():
    run = wandb.init(group="bernaise")
    # conf stores the modified configuration
    conf = run.config.as_dict()
    print('GETTING CURRENT RUN CONFIG')
    # find optimizer keys, convert back to dictionaries
    new_conf = {}
    for k in conf.keys():
        parts = k.split('.')
        if len(parts) == 2:
            if parts[0] not in new_conf:
                new_conf[parts[0]] = {}
            new_conf[parts[0]][parts[1]] = conf[k]
        if len(parts) == 3:
            if parts[0] not in new_conf:
                new_conf[parts[0]] = {}
            if parts[1] not in new_conf[parts[0]]:
                new_conf[parts[0]][parts[1]] = {}
            new_conf[parts[0]][parts[1]][parts[2]] = conf[k]
    # remove redundant string keys
    for k in [x for x in conf.keys() if '.' in x]:
        conf.pop(k, None)
    conf.pop('valid_set', None)
    # insert proper key-value pairs
    conf.update(new_conf)
    # write dummy yml
    print('WRITING DUMMY YAML')
    with open(dummy_yml, 'w') as f:
        yaml.safe_dump(conf, f, default_flow_style=False)
    # train
    print('TRAINING')
    print(conf)
    def cmd(args):
        with subprocess.Popen(args, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT) as process:
            for _line in process.stdout:
                line = _line.decode('utf8')
                print(line)
                if 'validation result' in line:
                    lm_loss = float(
                            [x for x in line.split('|') if \
                                    'lm_loss value' in x][0].split(':')[1].strip()
                            )
                    wandb.log({'lm_loss': lm_loss})
    cmd('python deepy.py train.py -d configs dummy.yml local_setup_neo.yml'.split(' '))
    run.finish()


sweep_id = wandb.sweep(
    sweep_config, project=project,
)
agent = wandb.agent(
    sweep_id=sweep_id,
    function=train,
    project=project
)
agent.run()
