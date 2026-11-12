"""Regenerate the measured-result section and trajectory plot from results.json."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
d = json.loads((root/'results.json').read_text(encoding='utf-8'))
path = d['two']['path']
fig, ax = plt.subplots(figsize=(6,6)); ax.set_xlim(-.5,4.5); ax.set_ylim(4.5,-.5); ax.set_xticks(range(5)); ax.set_yticks(range(5)); ax.grid()
for r,c in [(0,2),(1,2),(3,2),(4,2)]: ax.add_patch(plt.Rectangle((c-.5,r-.5),1,1,color='lightgray'))
for robot,color in [(0,'tab:blue'),(1,'tab:orange')]:
    positions = [p[robot] for p in path]
    ax.plot([p[1] for p in positions],[p[0] for p in positions],color=color,marker='o',alpha=.65,label='AB'[robot])
    for t,(r,c) in enumerate(positions):
        if t and positions[t] == positions[t-1]: continue
        end=t
        while end+1<len(positions) and positions[end+1]==positions[t]: end+=1
        ax.annotate(str(t) if end==t else f'{t}-{end}',(c,r),xytext=(4,8 if robot==0 else -12),textcoords='offset points',fontsize=7,color=color)
ax.legend(); ax.set_title('Two-robot greedy rollout (labels = time)'); fig.tight_layout(); fig.savefig(root/'two_trajectory.png',dpi=150); plt.close(fig)

table = '\n'.join(f'| {i} | {tuple(a)} | {tuple(b)} |' for i,(a,b) in enumerate(path))
section = f'''All four test groups passed. Experiments used Python 3.14.3, NumPy {d['versions']['numpy']}, PyTorch {d['versions']['torch']} and Matplotlib {matplotlib.__version__} on CPU. Run `python warehouse.py` from this directory to reproduce, followed by `python finish_report.py` to rebuild this result section. Raw episode logs and [results.json](results.json) accompany the report. Seeds 0–99 were used for the 100 random-policy episodes.

| Default start | Budget / seed | Result | Steps to goal | Invalid moves |
|---|---|---|---:|---:|
| Q-table | 1,500 episodes / 7 | Success | 4 | 0 |
| Single DQN | 12,000 steps / 7 | Success | 4 | 0 |
| Joint DQN | 60,000 steps / 7 | Timeout, no collision | — | 0 |

The random policy succeeded in 15/100 runs, timed out in 85/100, averaged 25.4 steps among successes, and made 970 invalid moves in total (9.7 per run). Both single-robot greedy policies followed the four-step shortest route, shown in [the trajectory plot](single_trajectories.png). [Training curves](q_training.png) use a 50-episode moving average; [training data](q_training.csv) includes epsilon, invalid moves and timeouts.

From `(0,0)`, both single policies succeeded in six steps with zero invalid moves. From A=`(0,0)`, B=`(4,4)`, the joint controller timed out at 40 steps without collision, with one invalid move. These are single deterministic rollouts without retraining, not independent trials.

The default joint run failed: A entered the passage first, reaching `(2,2)` at step 2. B moved aside through `(3,4)` and `(4,4)` to `(4,3)`, where it stayed while A reached its goal. B never completed its delivery. The [trajectory plot](two_trajectory.png) labels time steps; repeated positions are shown as ranges. Positions after every simultaneous step are below:

| Step | A | B |
|---:|---|---|
{table}
'''
report = root/'submission.md'
text = report.read_text(encoding='utf-8')
start = text.index('## 5. Experiments and discussion') + len('## 5. Experiments and discussion')
end = text.index('Training return can remain noisy', start)
report.write_text(text[:start]+'\n\n'+section+'\n'+text[end:],encoding='utf-8')
