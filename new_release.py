from py_wake.utils import git_utils
from py_wake.utils.git_utils import _run_git_cmd, git_path


MAJOR = 0
MINOR = 1
PATCH = 2


def get_tag(git_repo_path=None):
    return _run_git_cmd(['git', 'describe', '--tags', '--abbrev=0'], git_repo_path)


def set_tag(tag, push, git_repo_path=None):
    _run_git_cmd(["git", "tag", tag], git_repo_path)
    if push:
        _run_git_cmd(["git", "push"], git_repo_path)
        _run_git_cmd(["git", "push", "--tags"], git_repo_path)


def new_release(level=MINOR):
    """Make new tagged release"""

    if 'dirty' in git_utils.get_git_version(git_path):
        raise Exception("Commit before making new release")

    last_tag = get_tag(git_path)

    def next_tag(tag):
        return ".".join([str(int(n) + (0, 1)[i == level]) for i, n in enumerate(tag.split(".")[:2] + [0])])

    new_tag = next_tag(last_tag)

    print("Making tag:", new_tag)
    set_tag(new_tag, push=True, git_repo_path=git_path)

    print('Done', new_tag)


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 2:
        new_release(int(sys.argv[1]))
    else:
        new_release()
