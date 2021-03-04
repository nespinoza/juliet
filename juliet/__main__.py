import sys


def main(args=None):
    if args is None:
        print('ERROR: You need to pass flags (e.g., -lcfilename, -priorfile, etc.) for juliet to work. \n'+\
              ' Check out the wiki documentation for a list of flags at https://github.com/nespinoza/juliet/wiki/Installing-and-basic-usage.')
        args = sys.argv[1:]
    else:
        print(args)

    print("juliet command mode is under development. In the meantime, use the juliet.py script.")

    # Do argument parsing here (eg. with argparse) and anything else
    # you want your project to do.

if __name__ == "__main__":
    main()
