import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='t77-v0',
    entry_point='gym_T77.envs:T77Env',
)