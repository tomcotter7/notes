# Recommendation Systems

## Production Level Systems

### ByteDance's Monolith

[Paper](https://arxiv.org/pdf/2209.07663)

In the paper, ByteDance present "Monolith" a production level recommendation algorithm which is likely what powers TikTok. They talk about what it takes to get an embedding model based recommendation system to work at such high volume, but also update realtime to the users preferences.

Essentially, all actions along with the features upon which those actions happened are collected and stored in their training data. A training worker trains the recommendation model on the collected data, updating the 'Training Paramter Store' after each backward pass. Periodically, these parameters are syncronized to the 'Serving Paramter Store', which will recommend to the user the next relevant thing.

This synchronization happens in a different way for sparse and dense parameters. They found that sparse parameters dominate the size of the recommendation models, so they consistnely sync these parameters, while updating dense parameters less frequently.

It seems as if they give out 'embeddings' to IDs, which I can assume is people, and therefore can associate content embeddings to people embeddings in order to recommend certain content.

Fault - Tolerance is achieved by snapshotting the parameter store once per day. They foun d that this was a suitable tradeoff between computation overhead and model quality, since the model does not change significantly across a 24 hour period.
