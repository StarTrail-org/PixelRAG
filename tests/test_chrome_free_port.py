"""`free_port` hands out a usable port for --remote-debugging-port.

Both render backends used to derive the port from a fixed base (9400 in cdp,
12000 in fast_cdp via a process-local counter), so two pixelshot processes on
one machine picked the same ports: the second one's CDP client attached to the
first one's Chrome and the connection died when that run tore it down
(ConnectionResetError, or an empty result list).

Two calls are not asserted to differ — nothing guarantees the OS won't reuse a
port once the probe socket is closed. What matters is that the port is real and
free at the moment it is handed out.
"""

import socket

from pixelrag_render.chrome import free_port


def test_free_port_is_a_usable_port():
    port = free_port()
    assert isinstance(port, int)
    assert 1024 < port < 65536, f"not an ephemeral port: {port}"
    # Free at hand-out: Chrome must be able to bind it.
    with socket.socket() as s:
        s.bind(("127.0.0.1", port))


def test_free_port_does_not_return_the_old_fixed_bases():
    # Guards the regression, not the allocator: a reintroduced constant would
    # almost certainly be one of these.
    assert free_port() not in (9400, 12000)
