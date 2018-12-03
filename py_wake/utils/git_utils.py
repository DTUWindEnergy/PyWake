import os
import subprocess
import py_wake
git_path = os.path.dirname(py_wake.__file__) + "/../"


def _run_git_cmd(cmd, git_repo_path=None):
    git_repo_path = git_repo_path or os.getcwd()
    if not os.path.isdir(os.path.join(git_repo_path, ".git")):
        raise Warning("'%s' does not appear to be a Git repository." % git_repo_path)
    try:
        process = subprocess.Popen(cmd,
                                   stdout=subprocess.PIPE,
                                   universal_newlines=True,
                                   cwd=os.path.abspath(git_repo_path))
        stdout = process.communicate()[0]
        if process.returncode != 0:
            raise EnvironmentError()
        return stdout.strip()

    except EnvironmentError as e:
        raise Warning("Unable to run git\n%s" % str(e))


def get_git_version(git_repo_path=None):
    cmd = ["git", "describe", "--tags", "--dirty", "--always"]
    return _run_git_cmd(cmd, git_repo_path)


def update_git_version(module, git_repo_path=None):
    """Update <version_module>.__version__ to git version"""

    version_str = get_git_version(git_repo_path)
    assert os.path.isfile(module.__file__)
    with open(module.__file__, "w") as fid:
        fid.write("__version__ = '%s'\n" % version_str)

    # ensure file is written, closed and ready
    with open(module.__file__) as fid:
        fid.read()
    return version_str


def main():
    """Example of how to run (pytest-friendly)"""
    if __name__ == '__main__':
        update_git_version(py_wake, git_path)


main()
