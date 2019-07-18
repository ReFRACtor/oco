import os
import re
import glob

# Don't automatically import these modules, they may use C interface
# stuff and should not be available unless directly imported
NO_AUTO_IMPORT = ["__init__",]

for i in glob.glob(os.path.dirname(__file__) + "/*.py"):
    mname = os.path.basename(i).split('.')[0]
    # Don't automatically import test routines
    if(not re.match('.*_test', mname)) and (not mname in NO_AUTO_IMPORT):
        exec('from .%s import *' % mname)
