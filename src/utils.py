def get_obs_shape(observation_space):
    if observation_space.__class__.__name__ == 'Box':
        return observation_space.shape
    else:
        NotImplementedError

def get_state_shape(state_space):
    if state_space.__class__.__name__ == 'Box':
        return state_space.shape
    else:
        NotImplementedError

def get_act_shape(action_space):
    if action_space.__class__.__name__ == 'Discrete':
        return (action_space.n,)
    elif action_space.__class__.__name__ == 'Box':
        return action_space.shape
    else:
        NotImplementedError