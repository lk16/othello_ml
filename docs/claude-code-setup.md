# Project-isolated Claude Code with custom backend

How to configure Claude Code to use a custom API backend (e.g. DeepSeek, OpenRouter) for a specific project, while other projects use a different backend or an Anthropic subscription — all on the same machine. Works on **Linux** and **macOS**.

## How it works

1. **direnv** loads per-project environment variables from `.envrc` when entering the directory, and unloads them when leaving.
2. **`ANTHROPIC_API_KEY` + `ANTHROPIC_BASE_URL`** route API calls to the chosen backend.
3. **`CLAUDE_CONFIG_DIR`** pointed at a project-local directory isolates user-level config (history, sessions, credentials) from the global `~/.claude/`. Without OAuth credentials in that directory, Claude Code sees only the API key — no auth conflict warning.

## File layout

```
<project>/
├── .envrc                  # direnv: sets ANTHROPIC_* vars + CLAUDE_CONFIG_DIR
├── .claude-user/           # Isolated user config (gitignored)
│   ├── settings.json -> ~/.claude/settings.json   # symlink
│   └── plugins/     -> ~/.claude/plugins/          # symlink
├── .claude/                # Project-scoped settings (optional, can be committed)
│   └── settings.local.json
└── .gitignore              # Should ignore .envrc and .claude-user/
```

## Key env vars (in `.envrc`)

The backend-specific values below are an example (DeepSeek). Adjust `ANTHROPIC_BASE_URL`, the model names, and the API key for your provider.

```bash
export CLAUDE_CONFIG_DIR=$PWD/.claude-user          # Isolate user config
export ANTHROPIC_BASE_URL=<custom-backend-url>       # e.g. https://api.deepseek.com/anthropic
export ANTHROPIC_API_KEY=<your-api-key>
export ANTHROPIC_MODEL=<model-name>                  # e.g. deepseek-v4-pro[1m]
export ANTHROPIC_DEFAULT_OPUS_MODEL=<model-name>
export ANTHROPIC_DEFAULT_SONNET_MODEL=<model-name>
export ANTHROPIC_DEFAULT_HAIKU_MODEL=<model-name>
export CLAUDE_CODE_SUBAGENT_MODEL=<lighter-model>    # Used for subagents
export CLAUDE_CODE_EFFORT_LEVEL=max                  # Optional
```

## Setup steps

```bash
# 1. Create isolated config dir
mkdir -p .claude-user

# 2. Symlink global settings and plugins (preserve hooks, permissions, theme)
ln -sf ~/.claude/settings.json .claude-user/settings.json
ln -sf ~/.claude/plugins .claude-user/plugins

# 3. Add to .gitignore (skip if already present)
echo '.claude-user/' >> .gitignore
echo '.envrc' >> .gitignore

# 4. Add env vars to .envrc (see above), then allow direnv
direnv allow
```

## Auth precedence

| Priority | Auth method | Scope |
|----------|------------|-------|
| 1 (highest) | `ANTHROPIC_API_KEY` env var | Per-shell (direnv) |
| 2 | `/login` OAuth in `$CLAUDE_CONFIG_DIR/config.json` | Per `CLAUDE_CONFIG_DIR` |
| 3 (lowest) | `/login` OAuth in `~/.claude/config.json` | Global |

Setting `CLAUDE_CONFIG_DIR` to a directory without `config.json` removes tiers 2 and 3 — only the env var applies, so no auth conflict.

## Known issue

Claude Code shows an auth conflict warning when both `ANTHROPIC_API_KEY` and a `/login` session exist in the same config dir (tracked at [anthropics/claude-code#4733](https://github.com/anthropics/claude-code/issues/4733)). The `CLAUDE_CONFIG_DIR` approach sidesteps this by keeping the login session out of the active config dir.

## Platform notes

- **Linux** and **macOS** both store Claude Code config at `~/.claude/`. Symlinks and direnv work identically on both.
- **Windows** uses a different config path and has no native symlink or direnv support. This guide does not cover Windows.
