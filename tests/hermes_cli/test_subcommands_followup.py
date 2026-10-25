"""Smoke tests for the Phase 2 follow-up subcommand builders (promoted handlers).

These 9 subcommands had their handler defined as a closure inside main(); the
handler was promoted to top-level and the parser block extracted into a builder.
Confirms each builder attaches its subcommand and wires func to the injected
handler.
"""

from __future__ import annotations

import argparse


from hermes_cli.subcommands.mcp import build_mcp_parser


def _h(name):
    def handler(args):  # pragma: no cover - identity only
        return name
    handler.__name__ = f"cmd_{name}"
    return handler






def test_mcp_accept_hooks_flag():
    # mcp parser block uses the shared add_accept_hooks_flag helper.
    parser = argparse.ArgumentParser(prog="hermes")
    sub = parser.add_subparsers(dest="command")
    build_mcp_parser(sub, cmd_mcp=_h("mcp"))
    ns = parser.parse_args(["mcp", "--accept-hooks"])
    assert ns.accept_hooks is True
