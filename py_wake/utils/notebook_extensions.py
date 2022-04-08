def skip(line, cell=None):  # pragma: no cover
    '''Skips execution of the current line/cell if line evaluates to True.'''
    if "#" in line:
        line = line[:line.index('#')]
    print("Cell skipped. Precomputed result shown below. Remove '%%skip' to force run. ")
    if line.strip() == "" or eval(line):
        return

    get_ipython().ex(cell)


def load_ipython_extension(shell):  # pragma: no cover
    '''Registers the skip magic when the extension loads.'''
    shell.register_magic_function(skip, 'line_cell')


def unload_ipython_extension(shell):  # pragma: no cover
    '''Unregisters the skip magic when the extension unloads.'''
    del shell.magics_manager.magics['cell']['skip']
