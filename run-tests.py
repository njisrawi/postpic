#!/usr/bin/env python
#
# This file is part of postpic.
#
# postpic is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# postpic is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with postpic. If not, see <http://www.gnu.org/licenses/>.
#
# Copyright Stephan Kuschel, 2014-2015

# run all tests and pep8 verification of this project.
# It is HIGHLY RECOMMENDED to link it as a git pre-commit hook!
# Please see pre-commit for instructions.

# THIS FILE MUST RUN WITHOUT ERROR ON EVERY COMMIT!

import os
import subprocess


def exitonfailure(exitstatus, cmd=None):
    if exitstatus == 0:
        print('OK')
    else:
        print('run-tests.py failed. aborting.')
        if cmd is not None:
            print('The failing command was:')
            print(cmd)
        exit(exitstatus)

def run_autopep8(args):
        import autopep8
        if autopep8.__version__ < '1.2':
            print('upgrade to autopep8 >= 1.2 (installed: {:})'.format(str(autopep8.__version__)))
            exit(1)
        autopep8mode = '--in-place' if args.autopep8 == 'fix' else '--diff'
        argv = ['autopep8', '-r', 'postpic', '--ignore-local-config', autopep8mode, \
                '--ignore=W391,E123,E226,E24' ,'--max-line-length=99']
        print('===== running autopep8 =====')
        print('autopep8 version: ' + autopep8.__version__)
        print('$ ' + ' '.join(argv))
        autopep8.main(argv)

def run_alltests(python='python'):
    '''
    runs all tests on postpic. This function has to exit without error on every commit!
    '''
    python += ' '
    # make sure .pyx sources are up to date and compiled
    subprocess.call(python + os.path.join('.', 'setup.py develop --user'), shell=True)

    cmds = [python + '-m nose',
            python + '-m pep8 postpic --statistics --count --show-source '
            '--ignore=W391,E123,E226,E24 --max-line-length=99',
            python + os.path.join('examples', 'simpleexample.py'),
            python + os.path.join('examples', 'particleshapedemo.py'),
            python + os.path.join('examples', 'time_cythonfunctions.py')]
    for cmd in cmds:
        print('=====  running next command =====')
        print('$ ' + cmd)
        exitonfailure(subprocess.call(cmd, shell=True), cmd=cmd)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='''
        Without arguments this runs all tests
        on the postpic codebase.
        This script MUST EXIT WITHOUT ERROR on EVERY commit!
        ''')
    parser.add_argument('--autopep8', default='', nargs='?', metavar='fix',
                        help='''
        run "autopep8" on the codebase.
        Use "--autopep8" to preview changes. To apply them, use
        "--autopep8 fix"
        ''')
    pyversiongroup = parser.add_mutually_exclusive_group(required=False)
    pyversiongroup.add_argument('-2', action='store_const', dest='pycmd', const='python2',
                                help='Prefix all subcommands with "python2"',
                                default='python')
    pyversiongroup.add_argument('-3', action='store_const', dest='pycmd', const='python3',
                                help='Prefix all subcommands with "python3"')
    args = parser.parse_args()

    if args.autopep8 != '':
        run_autopep8(args)
        exit()

    run_alltests(args.pycmd)

if __name__ == '__main__':
    main()
