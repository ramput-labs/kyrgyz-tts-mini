import pytest

from tts_mini.cli import build_parser, main


@pytest.mark.parametrize(
    "argv",
    [
        ["speak", "салам", "-v", "man", "--rate", "1.2", "-p"],
        ["speak", "-f", "samples/texts.txt", "-o", "out.wav"],
        ["speak"],
        ["doctor", "--device", "cpu"],
    ],
)
def test_parser_accepts(argv):
    assert callable(build_parser().parse_args(argv).func)


def test_parser_rejects_unknown_voice():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["speak", "салам", "-v", "robot"])


def test_download_is_handed_to_the_downloader(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["download", "--help"])
    assert exit_info.value.code == 0
    assert "--pack" in capsys.readouterr().out


def test_download_rejects_unknown_model():
    with pytest.raises(SystemExit) as exit_info:
        main(["download", "robot"])
    assert exit_info.value.code == 2
