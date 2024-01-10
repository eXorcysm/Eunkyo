"""

This module implements functions to process GTP commands.

"""

class Response(object):
    def __init__(self, status, body):
        self.body    = body
        self.success = status

def bool_to_gtp(bool):
    """
    Convert Python boolean into GTP response.
    """

    if bool is True:
        return success("true")
    else:
        return success("false")

def error(command_body = ""):
    """
    Return unsuccessful GTP response.
    """

    return Response(status = False, body = command_body)

def serialize(gtp_command, gtp_response):
    """
    Serialize GTP response as string.
    """

    if gtp_command.sequence is None:
        command_sequence = ""
    else:
        command_sequence = str(gtp_command.sequence)

    if gtp_response.success:
        response_success = "="
    else:
        response_success = "?"

    return "{0}{1} {2}\n\n".format(response_success, command_sequence, gtp_response.body)

def success(command_body = ""):
    """
    Return successful GTP response.
    """

    return Response(status = True, body = command_body)
