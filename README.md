# rl_experiments
This repo contains code snippets that result from me trying to learn more about reinforcement learning. My eventual goal is to get to a point where I can train my Roomba to push a box to specific position and orientation, but before I get there, I need to touch up on some basics.

Examples in this repo (in order they were attempted):
## Cartpole
+ Usage: `python3 cartpole_learning_example.py`
+ Notes:
    + Additional attempts can be found in the archive folder
    + The implementation here uses vanilla Q-learning and trains in ~1 min, and works fairly well
        + To keep the Q-table small, all of the states are discretized into "buckets"
    + My primary modifications to this involved merging all of the visualizations I liked into one example to get a better understanding of what was happening
+ Lessons learned:
    + Deep-q learning takes a LOT longer than standard Q-learning (and is not always necessarily better), so discretizing the space if possible is always a good idea

## Navigation Example
+ Usage: `python3 nav_learning_example.py`
+ Notes:
    + Building off the cartpole example, this example contains a custom gym environment that creates a "simulated roomba" - this roomba moves according to a forward and angular velocity command. The simulation ends when the roomba either leaves the space or successfully navigates to the goal
    + Attempts to use Q-learning to navigate the roomba to the goal by discretizing the space, similar to how the cartpole did
    + Does train, and with significant smoothing (~100 episodes) the rewards do seem to improve with more training time, but seems to consistently cap at ~700 (when the goal reward is 1000) within 10000 iterations or so. The results don't look great. Could be improved (in theory), but I don't really understand what all the parameters do, so this is on hold and I will return when I have a better understanding of what's going on
+ Lessons learned:
    + How to create a custom environment
    + How to discretize a continuous space
    + Building the data pipeline from a custom input to training to testing

## Discrete Navigation Example
+ Usage: `python3 nav_learning_example.py`
+ Notes:
    + This is similar to the other discrete navigation example, except the environment is explicitly made to resemble a 2-D grid - the robot can only go up/down/left/right at 1 m/s, and the goal will also always occupy a 1x1 m space.
+ Current status: Q-table is not being updated properly? Needs more debugging, but a bit more foundational knowledge is needed to effectively debug. Will return

## Taxi Example
+ Usage: `python3 taxi_q_learning.py`
    + To view results: `asciinema play openaigym.video.0.264565.video000000.json`
+ Notes:
    + Trains in ~30 secs, example pulled straight from the tutorials available here https://simoninithomas.github.io/deep-rl-course/
    + Modified to use a custom "data saver" which is a cleaned-up version of the visualizations I'd used previously

## Atari
