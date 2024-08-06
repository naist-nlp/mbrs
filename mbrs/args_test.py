import pathlib

import yaml

from mbrs.cli.decode import get_argparser


class TestArgumentParser:
    def test_config_load(self, tmp_path: pathlib.Path):
        config_path = tmp_path / "config.yaml"
        hyps_path = tmp_path / "hyps.txt"
        cfg_dict = {
            "common": {
                "metric": "chrf",
                "decoder": "probabilistic_mbr",
                "num_candidates": 2,
            },
            "metric": {"char_order": 4},
            "decoder": {"reduction_factor": 16.0},
        }

        with open(config_path, mode="w") as f:
            yaml.dump(cfg_dict, f)

        with open(hyps_path, mode="w") as f:
            f.writelines(["tests", "a test"])

        cmd_args = [str(hyps_path), "--config_path", str(config_path)]
        parser = get_argparser(cmd_args)
        args = parser.parse_args(args=cmd_args)
        for key, value in cfg_dict["common"].items():
            assert getattr(args.common, key) == value
        for key, value in cfg_dict["metric"].items():
            assert getattr(args.metric, key) == value
        for key, value in cfg_dict["decoder"].items():
            assert getattr(args.decoder, key) == value
