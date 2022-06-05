from pettingzoo.mpe import simple_spread_v2
from pettingzoo.mpe import simple_v2
from pettingzoo.mpe import simple_adversary_v2
from pettingzoo.mpe import simple_crypto_v2
from pettingzoo.mpe import simple_push_v2
from pettingzoo.mpe import simple_speaker_listener_v3
from pettingzoo.mpe import simple_reference_v2
from pettingzoo.mpe import simple_tag_v2
from pettingzoo.mpe import simple_world_comm_v2

ENVS = {
    'simple_spread': simple_spread_v2,
    'simple': simple_v2,
    'simple_adversary': simple_adversary_v2,
    'simple_crypto': simple_crypto_v2,
    'simple_push': simple_push_v2,
    'simple_reference': simple_reference_v2,
    'simple_tag': simple_tag_v2,
    'simple_speaker_listener': simple_speaker_listener_v3,
    'simple_world_comm': simple_world_comm_v2
}