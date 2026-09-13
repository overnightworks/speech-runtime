"""Bootstrap VoxCPM after reserving stdout for its binary protocol."""

import argparse
import ctypes
import os
import signal


def main() -> int:
    """Redirect ordinary stdout, then enter the shared provider loop."""
    arguments = _arguments()
    protocol_stdout = os.dup(1)
    null = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, 1)
    os.close(null)
    try:
        install_parent_death_signal(arguments.expected_parent_pid)
    except RuntimeError:
        return 1
    from presentator_speech_provider_contract import (
        VOXCPM_SAMPLE_RATE,
        ProviderFunctions,
        serve_provider,
    )

    from presentator_voxcpm.model import load_model, pcm_chunks

    return serve_provider(
        device=arguments.device,
        cache=arguments.cache,
        protocol_stdout=protocol_stdout,
        sample_rate=VOXCPM_SAMPLE_RATE,
        functions=ProviderFunctions(load_model, pcm_chunks),
    )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-parent-pid", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--cache", required=True)
    return parser.parse_args()


def install_parent_death_signal(expected_parent_pid: int) -> None:
    """Kill this worker when its exact creator process ends."""
    if (
        ctypes.CDLL(None).prctl(1, signal.SIGKILL) != 0
        or os.getppid() != expected_parent_pid
    ):
        raise RuntimeError


if __name__ == "__main__":
    raise SystemExit(main())
