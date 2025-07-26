from typing import List, Any, Optional
import socket
import select
import io
import pickle
import subprocess
import os
import signal
from contextlib import contextmanager
import sys


@contextmanager
def Harvesting(verbose: bool=False) :

    try :
        host = os.environ['HARVESTER_HOST']
        port = int(os.environ['HARVESTER_PORT'])

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))

        yield lambda data : sock.sendall(pickle.dumps(data))

    except KeyboardInterrupt :
        if verbose :
            print(f'Worker on port {port} has been asked to stop')
    except (BrokenPipeError, ConnectionResetError) :
        if verbose :
            print(f'Worker on port {port} lost the connection with the harvester', file=sys.stderr)
    finally :
        sock.close()


class Harvester() :

    def __init__(self) :
        self._workers = []

    def number_of_workers(self) -> int :
        return len(self._workers)

    def working(self) -> bool :
        return len(self._workers) > 0

    def spawn_workers(self,
            bash_commands: List[str],
            harvester_ip: str='localhost',
            stdout: bool=False,
            stderr: bool=False,
            respawn: bool=False,
            verbose: bool=True
        ) -> None :

        spawn_args = {}
        spawn_args['harvester_ip'] = harvester_ip
        spawn_args['stdout'] = stdout
        spawn_args['stderr'] = stderr
        spawn_args['respawn'] = respawn
        spawn_args['verbose'] = verbose

        new_workers = []

        try :
            for command in bash_commands :

                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.bind((harvester_ip, 0))
                host, port = sock.getsockname()
                sock.listen(1)

                worker_env = os.environ.copy()
                worker_env['HARVESTER_HOST'] = host
                worker_env['HARVESTER_PORT'] = str(port)
                proc = subprocess.Popen(
                           command,
                           shell=True,
                           start_new_session=True,
                           executable='/bin/bash',
                           env=worker_env,
                           stdin=subprocess.DEVNULL,
                           stdout=(None if stdout else subprocess.DEVNULL),
                           stderr=(None if stderr else subprocess.DEVNULL)
                       )

                spawn_args['bash_commands'] = [command]

                new_workers.append({'proc': proc, 'socket': sock, 'port': port, 'spawn_args': spawn_args})

            for worker in new_workers :

                if verbose :
                    print(f'Waiting for connection... ', end='', flush=True)

                worker['socket'].settimeout(1)

                while True :
                    try :
                        conn, addr = worker['socket'].accept()
                    except TimeoutError :
                        if worker['proc'].poll() is not None :
                            raise RuntimeError(f'Worker {worker["proc"].pid} died before establishing the connection')
                    else :
                        break

                conn.setblocking(False)

                worker['connection'] = conn

                if verbose :
                    print(f'\rSocket {host}:{worker["port"]} connected to worker {worker["proc"].pid}', flush=True)
        except :
            self._stop_workers(new_workers)
            raise

        self._workers.extend(new_workers)

    def wait(self, timeout: Optional[float]=None) -> int :

        readable_sockets, _, _ = select.select([worker['connection'] for worker in self._workers], [], [], timeout)

        return len(readable_sockets)

    def harvest(self, bufsize: int=65536, verbose: bool=True) -> List[Any] :

        if not self._workers :
            return []

        harvest = []

        dead_workers = []

        for worker in self._workers :
            try :
                data_bytes = b''
                while True :
                    data_chunk = worker['connection'].recv(bufsize)
                    if not data_chunk : # Connection lost
                        if verbose :
                            print(f'Connection lost with worker {worker["proc"].pid}', file=sys.stderr)
                        dead_workers.append(worker)
                        break
                    data_bytes += data_chunk
            except BlockingIOError : # Nothing more to read
                pass

            if not data_bytes : # No data came from the worker but it is still connected
                continue

            data_io_stream = io.BytesIO(data_bytes)

            try :
                while True :
                    try :
                        data_object = pickle.load(data_io_stream)
                        harvest.extend(data_object if type(data_object) == list else [data_object])
                    except pickle.UnpicklingError :
                        if verbose :
                            print(f'Data received from worker {worker["proc"].pid} was truncated -- skipping this data', file=sys.stderr)
            except EOFError : # End of the data stream
                pass

        # Remove the dead workers from the list and spawn new ones if required to:
        for worker in dead_workers :
            if worker['spawn_args']['respawn'] :
                self.spawn_workers(**worker['spawn_args'])
            self._workers.remove(worker)

        return harvest

    def _stop_workers(self, workers) -> None :

        for worker in workers :
            if worker['proc'].poll() is None :
                os.killpg(worker['proc'].pid, signal.SIGINT)

    def stop_workers(self) -> None :

        self._stop_workers(self._workers)
