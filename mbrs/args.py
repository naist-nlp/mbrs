from __future__ import annotations

import argparse
import importlib
import logging
import os
import sys
from argparse import Namespace
from pathlib import Path
from typing import Sequence

import simple_parsing
from simple_parsing.wrappers.field_wrapper import ArgumentGenerationMode, FieldWrapper

logger = logging.getLogger(__name__)


class ArgumentParser(simple_parsing.ArgumentParser):
    IMPORTED_MODULES: set[Path] = set()

    def add_plugin_argumnets(self, parser: ArgumentParser) -> None:
        """Add arguments for plugins.

        Args:
            parser (ArgumentParser): Argument parser.
        """
        parser.add_argument(
            "--plugin_dir",
            type=Path,
            default=None,
            help="Path to a directory containing user defined plugins.",
        )

    def import_plugin(self, plugin_dir: Path) -> None:
        """Import plugin modules.

        Args:
            plugin_dir (pathlib.Path): A directory containing user defined plugins.
        """
        plugin_dir = plugin_dir.absolute()
        if not os.path.exists(plugin_dir) or not os.path.isdir(
            os.path.dirname(plugin_dir)
        ):
            raise FileNotFoundError(plugin_dir)

        module_parent, module_name = os.path.split(plugin_dir)
        if plugin_dir not in ArgumentParser.IMPORTED_MODULES:
            if module_name not in sys.modules:
                sys.path.insert(0, module_parent)
                importlib.import_module(module_name)
            elif plugin_dir in sys.modules[module_name].__path__:
                logger.info(f"--plugin_dir={plugin_dir} has already been imported.")
            else:
                raise ImportError(
                    f"Failed to import --plugin_dir={plugin_dir} because the module name "
                    f"({module_name}) is not globally unique."
                )
            self.IMPORTED_MODULES.add(plugin_dir)

    def preprocess_parser(self) -> None:
        """Preprocess ArgumentParser."""
        self.parse_known_args_preprocess(sys.argv[1:])

    def parse_known_args_preprocess(
        self,
        args: Sequence[str] | None = None,
        namespace: Namespace | None = None,
        attempt_to_reorder: bool = False,
    ) -> None:
        # default Namespace built from parser defaults
        if namespace is None:
            namespace = Namespace()
        if self.config_path:
            if isinstance(self.config_path, Path):
                config_paths = [self.config_path]
            else:
                config_paths = self.config_path
            for config_file in config_paths:
                self.set_defaults(config_file)

        if self.add_config_path_arg:
            temp_parser = ArgumentParser(
                add_config_path_arg=False,
                add_help=False,
                add_option_string_dash_variants=FieldWrapper.add_dash_variants,
                argument_generation_mode=FieldWrapper.argument_generation_mode,
                nested_mode=FieldWrapper.nested_mode,
            )
            temp_parser.add_argument(
                "--config_path",
                type=Path,
                default=self.config_path,
                help="Path to a config file containing default values to use.",
            )
            args_with_config_path, args = temp_parser.parse_known_args(args)
            config_path = args_with_config_path.config_path

            if config_path is not None:
                config_paths = (
                    config_path if isinstance(config_path, list) else [config_path]
                )
                for config_file in config_paths:
                    self.set_defaults(config_file)

            # Adding it here just so it shows up in the help message. The default will be set in
            # the help string.
            if self._option_string_actions.get("--config_path", None) is None:
                self.add_argument(
                    "--config_path",
                    type=Path,
                    default=config_path,
                    help="Path to a config file containing default values to use.",
                )

            # Plugin loader
            self.add_plugin_argumnets(temp_parser)
            args_with_plugin_dir, args = temp_parser.parse_known_args(args)
            plugin_dir: Path | None = args_with_plugin_dir.plugin_dir
            if plugin_dir is not None:
                self.import_plugin(plugin_dir)
            self.add_plugin_argumnets(self)

        assert isinstance(args, list)
        self._preprocessing(args=args, namespace=namespace)

    def parse_known_args(
        self,
        args: Sequence[str] | None = None,
        namespace: Namespace | None = None,
        attempt_to_reorder: bool = False,
    ):
        # NOTE: since the usual ArgumentParser.parse_args() calls
        # parse_known_args, we therefore just need to overload the
        # parse_known_args method to support both.
        if args is None:
            # args default to the system args
            args = sys.argv[1:]
        else:
            # make sure that args are mutable
            args = list(args)

        self.parse_known_args_preprocess(
            args=args, namespace=namespace, attempt_to_reorder=attempt_to_reorder
        )
        logger.debug(
            f"Parser {id(self)} is parsing args: {args}, namespace: {namespace}"
        )
        parsed_args, unparsed_args = super(
            simple_parsing.ArgumentParser, self
        ).parse_known_args(args, namespace)

        if unparsed_args and self._subparsers and attempt_to_reorder:
            logger.warning(
                f"Unparsed arguments when using subparsers. Will "
                f"attempt to automatically re-order the unparsed arguments "
                f"{unparsed_args}."
            )
            index_in_start = args.index(unparsed_args[0])
            # Simply 'cycle' the args to the right ordering.
            new_start_args = args[index_in_start:] + args[:index_in_start]
            parsed_args, unparsed_args = super(
                simple_parsing.ArgumentParser, self
            ).parse_known_args(new_start_args)

        parsed_args = self._postprocessing(parsed_args)
        return parsed_args, unparsed_args


class DataclassWrapper(simple_parsing.wrappers.DataclassWrapper):
    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        if self._name == "common":
            FieldWrapper.argument_generation_mode = ArgumentGenerationMode.FLAT
        else:
            FieldWrapper.argument_generation_mode = ArgumentGenerationMode.NESTED
        super().add_arguments(parser)
        FieldWrapper.argument_generation_mode = ArgumentGenerationMode.NESTED
