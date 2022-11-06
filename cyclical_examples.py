# examples of cyclical learning rate schedules

import os
import argparse
import matplotlib.pyplot as plt

import profiles

parser = argparse.ArgumentParser(description='Cyclical Learning Rate Schedules')
parser.add_argument('--save-dir', type=str, default='./figs')
parser.add_argument('--total-iter', type=int, default=1000)
parser.add_argument('--stepsize', type=int, default=250)
parser.add_argument('--max-lr', type=float, default=0.1)
parser.add_argument('--min-lr', type=float, default=0.001)
args = parser.parse_args()

profile2func = {
    'linear': [profiles.linear_growth, profiles.linear_decay],
    'cos': [profiles.cos_growth, profiles.cos_decay],
    'exp': [profiles.exp_growth, profiles.exp_decay],
    'rex': [profiles.rex_growth, profiles.rex_decay],
}

profile2sched = {}

for profile in profiles.profile_list:
    lr_schedule = []
    for _iter in range(args.total_iter + 1):
        curr_cycle = _iter // args.stepsize
        if (curr_cycle % 2) == 0:
            growth_profile = profile2func[profile][0]
            _lr = growth_profile(_iter, args.stepsize, args.min_lr, args.max_lr)
        else:
            decay_profile = profile2func[profile][1]
            _lr = decay_profile(_iter, args.stepsize, args.min_lr, args.max_lr)
        lr_schedule.append(_lr)
    profile2sched[profile] = lr_schedule

plt.figure(figsize=(8, 6))
plt.ylabel('Learning Rate', fontsize=16)
plt.xlabel('Training Iterations', fontsize=16)
for profile, lr_sched in profile2sched.items():
    plt.plot(lr_sched, label=profile, linewidth=2)
plt.legend(fontsize=12)
plt.savefig(os.path.join(args.save_dir, f'cyclical_lr_scheds.png'))
