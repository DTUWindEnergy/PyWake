import os
from py_wake.git_utils import get_tag

repo = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
version = get_tag(repo)[1:]

__version__ = version
__release__ = version
