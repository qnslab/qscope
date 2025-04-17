import click

from .smu import smu


@click.group()
def dev():
    """Hardware device control tools."""
    pass


dev.add_command(smu)
