# Monte-Carlo Policy Gradients

## Introduction

This md file is intended to be read in conjunction with the code. It'll serve as a reference containing more detail and visual explanations for what's happening at a technical level.


## Policy Gradient Theorem

We will begin with the derivation of the policy gradient theorem from first principles. To update our policy in a way that converges to optimality, we are interested in obtaining the policy gradient with respect to our parameters. For that purpose, we define an objective function $J$ as substitute for the true value function of a given policy. Since we want to maximize value, it makes sense to perform gradient ascent on this function.

$$J(\theta)=v_{\pi_\theta}(s_0)$$

Let's begin with the definition of the value function.

$$\begin{align*}v_{\pi_\theta}(s)&= \mathbb E_\pi[\sum_{k=0}^{\infty} \gamma^k R_{t+k}|S_t=s]\:\forall\:s\in S\\
&=\mathbb E_\pi[G_t|S_t=s]\end{align*}$$

Recall that this is just saying that the value of a policy $\pi_{\theta}$ with parameters $\theta$ is the expected value across all actions sampled from the policy and the infinite discounted return from following that policy thereafter. 

The value function can also be rewritten in terms of the Q-value.

$$v_{\pi_\theta}(s)=\sum_a\pi(a|s)\cdot q_\pi(s, a)$$

We want to derive the gradient, $\nabla v_\pi$. Applying product rule, we have:

$$\begin{align*}
\nabla v_{\pi_\theta}&=\nabla[\sum_a\pi(a|s)\cdot q_\pi(a, s)]\\
&= \sum_a[\nabla\pi(a|s)q_\pi(a, s)+\pi(a|s)\nabla q_\pi(a, s)]
\end{align*}$$

We can now expand the q-value function to its original definition:

$$\begin{align*}
&= \sum_a[\nabla\pi(a|s)q_\pi(a, s)+\pi(a|s)\nabla \sum_{s', r} p(s', r|s, a)(r+v_\pi(s'))]\\
\end{align*}$$

Since the reward is a constant, the gradient will not depend on it:

$$\begin{align*}
&= \sum_a[\nabla\pi(a|s)q_\pi(a, s)+\pi(a|s)\sum_{s'} p(s'| s, a)\nabla v_\pi(s')]\\
\end{align*}$$

Given that $s'$ is the next state, we can also unroll the value function for $v_\pi(s')$ as

$$\begin{align*}
&= \sum_a[\nabla\pi(a|s)q_\pi(a, s)+\pi(a|s)\sum_{s'} p(s'| s, a)\nabla \sum_{a'}\pi(a'|s')\cdot q(s',a')]\\
&= \sum_a[\nabla\pi(a|s)q_\pi(a, s)+\pi(a|s)\sum_{s'} p(s'| s, a)\\&\sum_{a'}[\nabla\pi(a'|s')\cdot q_\pi(s',a')+\pi(a'|s')\sum_{s''}p(s''|s',a')\nabla v_\pi(s'')]]\\
\end{align*}$$

In general, we can define this repeated unrolling as an infinite sum across states and steps:

$$\begin{align*}
&= \sum_{x\in S}\sum_{k=0}^{\infty}Pr(s\to x, k, \pi)\sum_a\nabla\pi(a|x)\cdot q_\pi(x, a)\\
\end{align*}$$

We define $Pr(s\to x, k, \pi)$ as the probability of reaching some state $x$ at timestep $k$ following policy $\pi_\theta$. 

Notes:
- Recall the definition of a infinite sum: by summing over k timesteps to infinity we account for the recursive nature of the q function
- Similarly, we loop over all states to account for all possible outputs of the value function
- The action probability and transition probabilitty are combined into a single probability function
- the first term $\sum_a\nabla\pi(a|s)\cdot q_\pi(s, a)$ is left untouched as our recursive base case

Finally, we can rewrite this by substituting $\eta$ for our probability function

$$\begin{align*}
&=\sum_s(\sum_{k=0}^{\infty}Pr(s_0\to s, k, \pi))\sum_a\nabla\pi(a|s)\cdot q_\pi(s, a)\\
&=\sum_s\eta(s)\sum_a\nabla\pi(a|s)\cdot q_\pi(s, a)\\
&=\sum_{s'}\eta(s')\sum_s\frac{\eta(s)}{\sum\limits_{s'}\eta(s')}\sum_a\nabla\pi(a|s)\cdot q_\pi(s, a)\\
&\propto \sum_s \mu(s) \sum_a\nabla\pi(a|s)\cdot q_\pi(s,a)
\end{align*}$$

