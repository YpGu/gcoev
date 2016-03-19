1. simulation: 
  co-evolution of (1) latent attributes and (2) graph
  (a) X(t) -> G(t) [generation]: the probability of a link is porpotional to the interaction of attributes of two endpoints
  (b) X(t), G(t) -> X(t+1) [propagation]: the value of latent attribute is sampled from the previous value and the graph:
      X(n+1) ~ p(X | X(n), G)

2. inference
  (a) inference_-1_1: Implementation of a basic version of paper
    "Dynamic Probabilistic Models for Latent Feature Propagation in Social Networks"
  (b) similar as (a)

3. separate
  each time stamp, estimate X(t) = argmax { p(X(t) | X(t-1), G(t-1), G(t)) }

4. lfpm
  Latent Feature Propagation Model (global estimation)


