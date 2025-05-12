import os

import grid2op

grid2op.change_local_dir(os.path.join(os.getcwd(), "envs"))

# env_name = "l2rpn_case14_sandbox"
# grid2op.make(env_name)

# env_name = "l2rpn_neurips_2020_track1_small"
# grid2op.make(env_name)

env_name = "l2rpn_neurips_2020_track1_large"
grid2op.make(env_name)

# env_name = "l2rpn_wcci_2022"
# grid2op.make(env_name)
