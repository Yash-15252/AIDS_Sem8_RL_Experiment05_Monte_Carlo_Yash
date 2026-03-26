# AIDS_Sem8_RL_Experiment_05_TD

## ***YASH KHAMKAR - 221A030***
## ***Temporal Difference Learning***

---

## Aim
Implement model-free Temporal Difference (TD) learning algorithms (Monte Carlo Control, SARSA, Q-Learning) on custom 4×4 GridWorld environment.

## Problem Statement
```
GridWorld: 4×4 discrete grid (start=(0,0), goal=(3,3))
• Actions: right, down, left, up (deterministic)
• Reward: +1 at goal, -0.1 elsewhere  
• Goal: Learn optimal policy π* reaching goal from any state
```

**Custom Environment**: GridWorld class with `reset()`, `step()` methods.

## Brief Theory
**Model-Free RL**: Learns optimal Q*(s,a) without environment model (P,R unknown).

**Monte Carlo Control** (First-Visit MC):
```
G_t ← Σ_{k=t}^T γ^{k-t}R_{t+1}     (episode return)
Q(s,a) ← average_G(s,a)           (over first visits)
π(s) ← ε-greedy(Q(s,·))
```

**SARSA** (On-policy TD(0)):
```
Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
```

**Q-Learning** (Off-policy TD(0)):
```
Q(s,a) ← Q(s,a) + α[r + γmax_a'Q(s',a') - Q(s,a)]
```

## Implementation Explanation
`RL_EXP_5.ipynb` implements:

```
1. GridWorld Environment (4×4)
   • Actions: [right(0,1), down(1,0), left(0,-1), up(-1,0)]
   • Deterministic transitions
   
2. Monte Carlo Control (Exploring Starts + ε-greedy)
   • First-visit returns averaging
   • 5000 episodes
   
3. SARSA (ε-greedy)
   • TD(0) updates with next action a'
   • α=0.1, ε=0.1, γ=0.9
   
4. Q-Learning (ε-greedy)
   • TD(0) updates with max Q(s',a')
   • α=0.1, ε=0.1, γ=0.9
   
5. Training & Sample Q-values
   • Q_mc, Q_sarsa, Q_qlearning comparison
```

## Results
```
Expected Q-table Structure: Q[(state), action] for 16 states × 4 actions
Sample Output:
Monte Carlo Q-values converge to optimal policy values
SARSA: On-policy → slightly conservative near boundaries  
Q-Learning: Off-policy → optimal max values
All recover right/down policy toward (3,3)
```

**Performance Comparison**:
```
✓ Monte Carlo: High variance, unbiased, requires full episodes
✓ SARSA: Lower variance, on-policy bias  
✓ Q-Learning: Off-policy, maximal values, boundary clipping
✓ Episodes: 5000 each (converged)
```

## Sample Output
```
Sample Q-values from Monte Carlo: {((0,0), 0): 0.72, ...}
Sample Q-values from SARSA: {((0,0), 1): 0.68, ...}  
Sample Q-values from Q-Learning: {((0,0), 1): 0.71, ...}
```

## Conclusion
 **Model-Free TD Verified**: MC/SARSA/Q-Learning solve GridWorld optimally  
 **Algorithm Trade-offs**:
   - MC: Complete episodes required, unbiased
   - SARSA: Safe on-policy behavior  
   - Q-Learning: Aggressive off-policy optimality  
 **Convergence**: All achieve Q ≈ optimal values after 5000 episodes  
 **Custom Gym**: Clean environment for RL experimentation  
 **Foundation**: TD methods enable model-free deep RL (DQN etc.)

## References
1. Sutton & Barto, "RL: An Introduction" (Ch. 5: Monte Carlo, Ch. 6: TD Learning)
2. AIDS Sem8 RL Course Materials

## Setup & Run
```bash
cd AIDS_Sem8_RL_Experiment05_TD
pip install -r requirements.txt
jupyter notebook RL_EXP_5.ipynb
```

**Requirements**:
```
numpy
```

---

