# Superhermes version banner (funkylang/hermes fork).
#
# Convention (shared with fllama FLLAMA_VERSION and llama.cpp FORK_VERSION):
#   "<timestamp> <commit-subject-first-line>"
# Updated on every fork commit with the ACTUAL current time at update (strict ordering);
# printed in the startup banner and `hermes --version` so a deployed tree — which has
# no .git, being content-based — can identify the exact build it runs.

FORK_VERSION = "2026-11-22 17:11:26 Add hermes-agents start subcommand"
