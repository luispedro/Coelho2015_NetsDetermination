def mkdir_p(dirname):
    from os import makedirs
    try:
        makedirs(dirname)
    except:
        pass

