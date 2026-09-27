from .fql_agent import FQLAgent
from .iql_agent import IQLAgent
from .sacbc_agent import SACBCAgent
from .qam_agent import QAMAgent

agents = {
    "fql": FQLAgent,
    "iql": IQLAgent,
    "sacbc": SACBCAgent,
    "qam": QAMAgent,
}
