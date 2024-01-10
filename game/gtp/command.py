"""

This module implements the Go Text Protocol (GTP) command. GTP commands consist of:

- an optional sequence number used for matching responses with commands
- a command name
- one or more arguments to the command

Ex. "666, 'play', ('white', 'D7')"

"""

class Command(object):
    def __init__(self, sequence, name, args):
        self.sequence = sequence
        self.name     = name
        self.args     = tuple(args)

    def __eq__(self, other):
        return self.sequence == other.sequence and \
               self.name == other.name and         \
               self.args == other.args

    def __repr__(self):
        return "Command(%r, %r, %r)" % (self.sequence, self.name, self.args)

    def __str__(self):
        return repr(self)

def parse(command):
    pieces = command.split()  # used to check for sequence number

    try:
        sequence = int(pieces[0])
        pieces   = pieces[1:]
    except ValueError:
        sequence = None  # non-numeric input means no sequence number given

    return Command(sequence, pieces[0], pieces[1:])
