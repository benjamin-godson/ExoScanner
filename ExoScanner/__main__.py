# This is the main file of this program. It contains the main function which
# calls the run function with all the parameters it extracted from the program-
# call.

from ExoScanner.run import run

import sys

def main():
    run('../testdata') # TODO Change back to sys.argv[0]


if __name__ == "__main__":
    main()
