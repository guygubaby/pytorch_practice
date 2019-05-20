[DQN](https://www.jianshu.com/p/72cab5460ebe)

learn how to play games

1. Q-Learning 和 深度学习回顾

- Q-learning是通过不停地探索和更新Q表中的Q值从而计算出机器人行动的最佳路径的，公式为
- Q(s0,a2)新=Q(a0,a2) 旧 + α* [Q(s0,a2)目标 - Q(s0,a2)旧]
Q(s0,a2)目标 =R(s1) + γ*max Q(s1,a)
- 深度学习就是用神经网络来学习数据，常见的深度学习网络如全连接的，CNN，RNN等等
