# examples of using profiles and sampling rates to produce different LR schedules

import os
import argparse
import matplotlib.pyplot as plt

import profiles

parser = argparse.ArgumentParser(description='Cyclical Learning Rate Schedules')
parser.add_argument('--save-dir', type=str, default='./figs')
parser.add_argument('--total-iter', type=int, default=1000)
parser.add_argument('--profile', type=str, choices=profiles.profile_list, default='linear')
parser.add_argument('--max-lr', type=float, default=0.1)
parser.add_argument('--min-lr', type=float, default=0.001)
args = parser.parse_args()


sampling_rates = [500, 250, 100, 50, 1]
profile2func = {
    'linear': profiles.linear_decay,
    'cos': profiles.cos_decay,
    'exp': profiles.exp_decay,
    'rex': profiles.rex_decay,
}
sr2sched = {}

for sr in sampling_rates:
    lr_schedule = []
    tmp_lr = args.max_lr
    for _iter in range(args.total_iter):
        if _iter > 0 and (_iter % sr) == 0:
            decay_profile = profile2func[args.profile]
            tmp_lr = decay_profile(_iter, args.total_iter, args.min_lr, args.max_lr)
        lr_schedule.append(tmp_lr)
    sr2sched[sr] = lr_schedule

plt.figure(figsize=(8, 6))
plt.ylabel('Learning Rate', fontsize=16)
plt.xlabel('Training Iterations', fontsize=16)
for sr, lr_sched in sr2sched.items():
    plt.plot(lr_sched, label=f'Samp. Rate = {sr} iters', linewidth=2)
plt.legend(fontsize=12)
plt.savefig(os.path.join(args.save_dir, f'{args.profile}_scheds_with_samprate.png'))
