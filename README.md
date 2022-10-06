# Stateful Active Facilitator: Coordination and Environmental Heterogeneity in Cooperative Multi-Agent Reinforcement Learning

## Starcraft Installation
[SMAC](https://github.com/oxwhirl/smac) is a gym environment based on [Stracraft II Machine Learning API](https://github.com/Blizzard/s2client-proto) and Deepmind's [PySC2](https://github.com/deepmind/pysc2) that was developped by [WhiRL](http://whirl.cs.ox.ac.uk/).

### SMAC
First, start by installing SMAC:

```
pip install git+https://github.com/oxwhirl/smac.git
```
### StarCraft II
Next download [StarCraft II](https://github.com/Blizzard/s2client-proto#downloads). Here we'll download the latest version ``SC2.4.10`` and unzip it:

```
wget https://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip
unzip -P iagreetotheeula SC2.4.10.zip
```
Now, we save the path to ``StarCraftII`` as an environment variable ``$SC2PATH``:

```
echo "export SC2PATH=~/StarCraftII/" > ~/.bashrc
```
### SMAC Maps
We're almost there! We just need to donwload the SMAC maps and put them in  ``$SC2PATH/Maps/``:

```
cd $SC2PATH/Maps/
wget https://github.com/oxwhirl/smac/releases/download/v0.1-beta1/SMAC_Maps.zip
unzip SMAC_Maps.zip
rm -rf SMAC_Maps.zip
```

To test if everything is working as expected, run the following command to view the SMAC maps:

```
python -m smac.bin.map_list
```
Which should give the following output:

```
Name            Agents  Enemies Limit  
3m              3       3       60     
8m              8       8       120    
25m             25      25      150    
5m_vs_6m        5       6       70     
8m_vs_9m        8       9       120    
10m_vs_11m      10      11      150    
27m_vs_30m      27      30      180    
MMM             10      10      150    
MMM2            10      12      180    
2s3z            5       5       120    
3s5z            8       8       150    
3s5z_vs_3s6z    8       9       170    
3s_vs_3z        3       3       150    
3s_vs_4z        3       4       200    
3s_vs_5z        3       5       250    
1c3s5z          9       9       180    
2m_vs_1z        2       1       150    
corridor        6       24      400    
6h_vs_8z        6       8       150    
2s_vs_1sc       2       1       300    
so_many_baneling 7       32      100    
bane_vs_bane    24      24      200    
2c_vs_64zg      2       64      400    
```
To make sure everything is properly installed, run the following command:

```
python -m smac.examples.random_agents
```

Now, you're ready to run experiments on SMAC maps!

These instructions are based on the [original repository](https://github.com/oxwhirl/smac) and this [repository](https://github.com/marlbenchmark/on-policy).
## Comet Configuration
[comet.ml](https://www.comet.com/site/) is a great tool for tracking and logging experiments as well as running hyperparameter sweeps.

In order to get started, make sure you [create an account](https://www.comet.com/signup) on their website (you can sign up using your github account!). Once that is done, you'll receive your ``API_KEY``.

Next, install ``comet`` using the following command:

```
pip install comet-ml
```

Next, go to your ``$HOME`` folder an create a file named ``.comet.config`` as follows (this works on Linux):

```
touch .comet.config
```
Next, open your config file and start editing:

```
nano .comet.config
```

Finally, copy-paste the following:

```
[comet]
api_key=API_KEY
workspace=WORKSPACE
```

Now, you can kick-start a comet.ml experiment as follows:

```
from comet_ml import Experiment

experiment = Experiment(project_name="pytorch") # No need to explicitly provide the API_KEY as .comet.config has it already
```

For more information, you can check the [documentation](https://www.comet.com/docs/python-sdk/pytorch/).
