#!/usr/bin/env python
"""
This is a small script that does the equivalent of multicrab.
"""
import os
from optparse import OptionParser

from CRABAPI.RawCommand import crabCommand
from CRABClient.ClientExceptions import ClientException
from httplib import HTTPException


def getOptions():
    """
    Parse and return the arguments provided by the user.
    """
    usage = ("Usage: %prog --crabCmd CMD [--workArea WAD --crabCmdOpts OPTS]"
             "\nThe multicrab command executes 'crab CMD OPTS' for each project directory contained in WAD"
             "\nUse multicrab -h for help")

    parser = OptionParser(usage=usage)

    parser.add_option('-c', '--crabCmd',
                      dest = 'crabCmd',
                      default = '',
                      help = "The crab command you want to execute for each task in DIR",
                      metavar = 'CMD')

    parser.add_option('-w', '--workArea',
                      dest = 'workArea',
                      default = '',
                      help = "work area directory (only if CMD != 'submit')",
                      metavar = 'WAD')

    parser.add_option('-o', '--crabCmdOpts',
                      dest = 'crabCmdOpts',
                      default = '',
                      help = "options for crab command CMD",
                      metavar = 'OPTS')

    (options, arguments) = parser.parse_args()

    if arguments:
        parser.error("Found positional argument(s): %s." % (arguments))
    if not options.crabCmd:
        parser.error("(-c CMD, --crabCmd=CMD) option not provided.")
    if options.crabCmd != 'submit':
        if not options.workArea:
            parser.error("(-w WAR, --workArea=WAR) option not provided.")
        if not os.path.isdir(options.workArea):
            parser.error("'%s' is not a valid directory." % (options.workArea))

    return options


def main():
    """
    Main
    """
    options = getOptions()

    # If you want crabCommand to be quiet:
    #from CRABClient.UserUtilities import setConsoleLogLevel
    #from CRABClient.ClientUtilities import LOGLEVEL_MUTE
    #setConsoleLogLevel(LOGLEVEL_MUTE)
    # With this function you can change the console log level at any time.

    # To retrieve the current crabCommand console log level:
    #from CRABClient.UserUtilities import getConsoleLogLevel
    #crabConsoleLogLevel = getConsoleLogLevel()

    # If you want to retrieve the CRAB loggers:
    #from CRABClient.UserUtilities import getLoggers
    #crabLoggers = getLoggers()

    # Execute the command with its arguments for each directory inside the work area.
    for dir in os.listdir(options.workArea):
        projDir = os.path.join(options.workArea, dir)
        if not os.path.isdir(projDir):
            continue
        # Execute the crab command.
        msg = "Executing (the equivalent of): crab %s --dir %s %s" % (options.crabCmd, projDir, options.crabCmdOpts)
        print "-"*len(msg)
        print msg
        print "-"*len(msg)
        try:
            crabCommand(options.crabCmd, dir = projDir, *options.crabCmdOpts.split())
        except HTTPException as hte:
            print "Failed executing command %s for task %s: %s" % (options.crabCmd, projDir, hte.headers)
        except ClientException as cle:
            print "Failed executing command %s for task %s: %s" % (options.crabCmd, projDir, cle)


if __name__ == '__main__':
    main()
