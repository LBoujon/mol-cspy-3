from cspy.apps.threshold_utils import ThreshUtilsCLI
import pytest


def test_app(capsys) -> None:
    commands = [func for func in dir(ThreshUtilsCLI) if callable(getattr(ThreshUtilsCLI, func)) and not func.startswith("__")]

    for command in commands:
        with pytest.raises(SystemExit) as sys_exit:
            # dumb test to catch that the app actually runs
            ThreshUtilsCLI([command, "--help"])
        assert sys_exit.value.code == 0

        screen_output = capsys.readouterr().out
        assert "usage:" in screen_output
        assert command in screen_output