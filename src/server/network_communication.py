import socket


class ACPCNetworkCommunication(object):

    connection: socket

    def __init__(self):
        pass

    # --- Connects over a network socket.
    # --
    # -- @param server the server that sends states to DyypHoldem, and to which
    # -- DyypHoldem sends actions
    # -- @param port the port to connect on
    def connect(self, server, port):
        self.connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connection.connect((server, int(port)))
        self._handshake()

    # --- Sends a handshake message to initialize network communication.
    # -- @local
    def _handshake(self):
        self.send_line('VERSION:2.0.0')

    # --- Sends a message to the server.
    # -- @param line a string to send to the server
    def send_line(self, line):
        self.connection.send((line + "\r\n").encode())

    # --- Waits for a text message from the server. Blocks until a message is
    # -- received.
    # -- @return the message received
    def get_line(self, size=1024):
        # peek into receive buffer and only receive one line at a time
        line = self.connection.recv(size, socket.MSG_PEEK)
        eol = line.find(b'\n')
        if eol >= 0:
            size = eol + 1
        else:
            size = len(line)
        out = self.connection.recv(size).decode()
        return out

    # --- Ends the network communication.
    def close(self):
        self.connection.close()
