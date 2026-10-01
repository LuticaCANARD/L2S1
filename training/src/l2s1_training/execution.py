"""Bounded child processes with durable stage records."""
import os
import signal
import subprocess
import time


def stop_process(process):
    if os.name == 'nt':
        subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait(timeout=15)


def execute_stage(name, command, out, env, status, persist, timeout):
    command = [str(x) for x in command]
    record = dict(name=name, command=command, status='running', log=f'{name}.log')
    status['stages'].append(record)
    persist()
    started = time.monotonic()
    process = None
    try:
        with (out/f'{name}.log').open('x', encoding='utf-8') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env,
                                       start_new_session=os.name != 'nt')
            record['exit_code'] = process.wait(timeout=timeout)
        record['status'] = 'ok' if record['exit_code'] == 0 else 'failed'
        if record['exit_code']:
            raise RuntimeError(f'{name} failed; see {out / record["log"]}')
    except subprocess.TimeoutExpired:
        record['status'] = 'timeout'
        raise TimeoutError(f'{name} exceeded {timeout:g}s; see {out / record["log"]}') from None
    except KeyboardInterrupt:
        record['status'] = 'cancelled'
        raise
    except Exception as error:
        record.update(status='failed', error=str(error))
        raise
    finally:
        if process is not None:
            if process.poll() is None or record['status'] in ('timeout', 'cancelled'):
                stop_process(process)
            record['exit_code'] = process.returncode
        record['elapsed_s'] = time.monotonic()-started
        persist()
