import json

from qeclang.cli import main


def test_commands(capsys, tmp_path):
    assert main(["inspect"]) == 0
    assert json.loads(capsys.readouterr().out)["k"] == 1
    assert main(["distance", "--code", "five"]) == 0
    assert json.loads(capsys.readouterr().out)["upper_bound"] == 3
    path = tmp_path / "steane.json"
    assert main(["export", str(path)]) == 0
    assert main(["inspect", "--input", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["n"] == 7
    assert main(["search", "--trials", "2"]) == 0
    assert len(capsys.readouterr().out.splitlines()) == 2
    assert main(["simulate", "--trials", "10", "--probability", "0"]) == 0
    assert json.loads(capsys.readouterr().out)["trials"] == 10
    assert main(["simulate", "--trials", "10", "--decoder", "bp-osd0"]) == 0
    assert "min_sum" in json.loads(capsys.readouterr().out)["decoder_name"]
    assert main(["distance", "--max-weight", "-1"]) == 2
    assert "max_weight" in capsys.readouterr().err


def test_study_commands(capsys, tmp_path):
    database = str(tmp_path / "study.sqlite3")
    arguments = ["study", database, "--lift", "3", "--terms", "2", "--seed", "7"]
    assert main(arguments + ["--trials", "2"]) == 0
    assert json.loads(capsys.readouterr().out)["trials"] == 2
    assert main(arguments + ["--trials", "4"]) == 0
    assert json.loads(capsys.readouterr().out)["trials"] == 4
    assert main(["study-results", database]) == 0
    assert json.loads(capsys.readouterr().out)["trials"] == 4


def test_portfolio_and_zero_noise(capsys):
    assert main(["distance", "--method", "portfolio"]) == 0
    results = json.loads(capsys.readouterr().out)
    assert results["x"]["status"] == "exact"
    assert results["x"]["lower_bound"] == results["x"]["upper_bound"] == 3
    assert main(["distance", "--code", "five", "--method", "portfolio"]) == 2
    assert "CSS" in capsys.readouterr().err
    assert main(["distance", "--milp"]) == 2
    assert "portfolio" in capsys.readouterr().err
    assert main(["simulate", "--decoder", "bp", "--probability", "0", "--trials", "10"]) == 0
    assert json.loads(capsys.readouterr().out)["failures"] == 0


def test_study_cli_accepts_mixed_backend_limits(capsys, tmp_path):
    database = str(tmp_path / "mixed.sqlite3")
    args = ["study", database, "--lift", "4", "--terms", "2", "--trials", "1",
            "--backend", "mixed-abelian", "--max-component-axis", "1"]
    assert main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["trials"] == 1 and not result["pareto"]
    assert main(["study-results", database]) == 0
    assert json.loads(capsys.readouterr().out)["trials"] == 1
