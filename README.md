# Stateful Active Facilitator: Coordination and Environmental Heterogeneity in Cooperative Multi-Agent Reinforcement Learning

This is the official repository to the paper ["Stateful Active Facilitator: Coordination and Environmental Heterogeneity in Cooperative Multi-Agent Reinforcement Learning"](https://arxiv.org/abs/2210.03022)

In this paper, we tackle the coordination and environmental heterogeneity characteristics that are present in real-life scenarios. To be able to understand how existing MARL algorithms fare in environments with high levels of coordination and/or environmental heterogeneity, we introduce a suite of environments called **HECOGrid**, where users can manually tune the level of coordination and environmental heterogeneity in the provided environments.

To tackle the difficulty in learning that comes with high levels of coordination and/or environmental heterogeneity, we introduce a new model: the **Stateful Active Facilitator** which has a differentiable communication channel that allows agents to efficiently communicate during training to improve coordination, as well as a pool of policies that they can choose from in order to be resilient to increasing levels of environmental heterogeneity
![Stateful Active Faciliator](/assets/saf.jpg "Stateful Active Faciliator")
![HECOGrid](/assets/hecogrid.jpg "HECOGrid")

# Setup
Start by installing the required modules:
```
pip install -r requirements.txt
```
Next, install the marlgrid environment by executing the following lines:
```
cd src/envs/marlgrid_env/
pip install -e .
```

# HECOGrid Environment
The code and installation guide for HECOGrid can be found [here](https://github.com/veds12/hecogrid).

# Experiments
You can find scripts to run experiments in the folder ``scripts``. It is structured as follows:
```
scripts/
├── jobs_master_keyfortreasure.sh
├── jobs_master_OOD.sh
├── jobs_master_teamsupportenv.sh
├── jobs_master_teamtogetherenv.sh
├── OOD_tests
│   ├── OOD_coordination_test_compoundenv.sh
│   ├── OOD_coordination_test.sh
│   ├── OOD_heco_test.sh
│   ├── OOD_heterogeneity_test_compoundenv.sh
│   ├── OOD_heterogeneity_test.sh
│   ├── OOD_num_agents_test.sh
│   ├── OOD_num_treasures_test.sh
│   └── OOD_size_env_test.sh
├── testing.sh
└── training
    ├── baselines.sh
    └── saf.sh
```
In order to reproduce experimental results in the paper, you can run the appropriate ``jobs_master`` for the desired environment. For example, for the ``KeyForTreasure``, that would be ``jobs_master_keyfortreasure.sh``.

If you want to run our algorithm on your environments, you can check the structure of ``saf.sh`` in ``scripts/training`` folder.
# Comet Configuration
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

# Citation

To cite this project, please use:

```
@article{Liu2022StatefulAF,
  title={Stateful active facilitator: Coordination and Environmental Heterogeneity in Cooperative Multi-Agent Reinforcement Learning},
  author={Dianbo Liu and Vedant Shah and Oussama Boussif and Cristian Meo and Anirudh Goyal and Tianmin Shu and Michael Curtis Mozer and Nicolas Manfred Otto Heess and Yoshua Bengio},
  journal={ArXiv},
  year={2022},
  volume={abs/2210.03022}
}
```
