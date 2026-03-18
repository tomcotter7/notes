# Entropy

## Shannon Entropy

The entropy of a random variable quantifies the average level of uncertainty or information associated with the variable's possible outcome. Essentially, it measures the expected amount of information needed to describe the state of the variable.

$H(X) = -\sum_{x \in X} p(x) log p(x)$

so $x$ is all possible outcomes of the random variable $X$ and $p(x)$ is the probability of $x$.

Shannon entropy uses a log of base 2.

## Information Content

The information content of an event (surprisal) is a function that increases as the probability of the event decreases.

Entropy measures the average amount of information conveyed by identifying the outcome of a random variable.

## The First Law of Complexodynamics

There is an interesting blog post called [The First Law of Complexodynamics](https://scottaaronson.blog/?p=762) by Scott Aaronson. As context, the second law is "the entropy of any closed system tends to increase with time until it reaches a maximum value."

Aaronson proposes a new law, in which the same closed system has "complexity", which increases to a point, and then decreases. He tries to define complexity using a notion called "sophistication" from Kolmogorov complexity. Kolmogorov complexity of a string x is the length of the shortest computer program that outputs x.

For example, "123" * N can be written with a much shorter program than an N-length string. Therefore, the "123" * N has a smaller Kolmogorov complexity. However, Aaronson points out "A uniformly random string, he said, has close-to-maximal Kolmogorov complexity, but it’s also one of the least “complex” or “interesting” strings imaginable.  After all, we can describe essentially everything you’d ever want to know about the string by saying “it’s random”!". So, it actually is not that complex.

Formally, this is defined as:

- Given a set S of n-bit strings, let K(S) be the number of bits in the shortest computer program that outputs the elements of S and then halts. Also, given such a set S and an element x of S, let K(x|S) be the length of the shortest program that outputs x, given an oracle for testing membership in S.  Then we can let the sophistication of x, or Soph(x), be the smallest possible value of K(S), over all sets S such that:
    - x∈S and
    - K(x|S) ≥ log2(|S|) – c, for some constant c.  (In other words, one can distill all the “nonrandom” information in x just by saying that x belongs that S.)

log2(|S|) is the number of bits needed to distinguish a specific item out of all the items in set S (1 bit can distinguish 2 items - 0 or 1, 2 bits can distinguish 4 (00, 01, 10, 11), and so on). If K(x|S) is sig. lower than log2(|S|) this would mean that x is easier to describe than other members of S, meaning S wasn't the "tightest" possible description of x's structure (so K(x|S) has to be larger than log2(|S|))

Intuitively, Soph(X) is the length of the shortest computer program that describes, not necessarily x itself, but a set S of which x is a "random" or "generic" member. To illustrate, any string x with small Kolmogorov compleixty has small sophistication, since we can let S be the singleton set {x}. However, a uniformly random string also has small sophistication, since we can let S be the set ${0, 1}^{n}$ of all n-bit strings.

(This is because to describe the set of all $n$-bit strings, your program is just: Print all binary combinations of length n.)

Aaronson defines complextropy of an n-bit string x as:

"the number of bits in the shortest computer program that runs in n log(n) time, and that outputs a nearly-uniform sample from a set S such that (i) x∈S, and (ii) any computer program that outputs x in n log(n) time, given an oracle that provides independent, uniform samples from S, has at least log2(|S|)-c bits, for some constant c."

The key distinction from sophistication is the n log(n) time bound — sophistication is purely information-theoretic, complextropy is computationally bounded.

He leaves this definition unproven, but, in my opinion, I think that Ilya put this on his "30 Foundational Papers" because it describes the goal of ML/DL, which is "To increase the entropy of a system to the point at which the complextropy is at it's largest (so therefore it's the most interesting)".



