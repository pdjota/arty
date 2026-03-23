# Development setup

## Runtimes (asdf)

The project uses [asdf](https://asdf-vm.com/) for runtimes. Versions are pinned in [`.tool-versions`](../.tool-versions).

```bash
# Install the plugin and runtime if needed
asdf plugin add python
asdf install   # installs versions listed in .tool-versions
```

## Python env

Created a virtualenv and install dependencies:

```bash
python -m venv .venv
. .venv/bin/activate 
pip install -r requirements.txt
```
