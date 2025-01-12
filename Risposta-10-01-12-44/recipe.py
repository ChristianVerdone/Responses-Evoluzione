import os
import sys
import logging
from pathlib import Path

import click
from jinja2 import Environment, FileSystemLoader

_logger = logging.getLogger(__name__)


@click.group()
def cli():
    """
    Utility per generare e gestire una recipe MLflow.
    """
    logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")


@cli.command()
@click.option(
    "--recipe_name",
    default="nomedellarecipe",
    required=True,
    help="Nome della recipe da creare. Scegli tra 'classification' o 'regression'",
)
@click.option(
    "--recipe_root",
    default=".",
    required=False,
    help="Root directory in cui creare la sottocartella con la recipe",
)
def create(recipe_name, recipe_root):
    """
    Crea una nuova recipe MLflow.
    """

    if recipe_name not in ["classification", "regression"]:
        _logger.error("Il nome della recipe deve essere 'classification' o 'regression'")
        sys.exit(1)

    recipe_name = recipe_name + "/v1"

    recipe_dir = os.path.join(recipe_root, recipe_name)
    recipe_steps_dir = os.path.join(recipe_dir, "steps")

    # Crea la struttura delle directory
    os.makedirs(recipe_dir, exist_ok=True)
    os.makedirs(recipe_steps_dir, exist_ok=True)

    env = Environment(
        loader=FileSystemLoader(searchpath=os.path.join(os.path.dirname(__file__), "templates")),
        trim_blocks=True,
        lstrip_blocks=True,
    )

    # Crea i file dalla template
    # Recipe file
    template = env.get_template("recipe.yaml.jinja")
    with open(os.path.join(recipe_dir, "recipe.yaml"), "w") as f:
        f.write(template.render(recipe_name=recipe_name))

    # Local file
    template = env.get_template("local.yaml.jinja")
    with open(os.path.join(recipe_dir, "local.yaml"), "w") as f:
        f.write(template.render())

    # Steps directory
    for step in ["ingest", "split", "transform", "train", "evaluate"]:
        template = env.get_template(f"steps_{step}.py.jinja")
        with open(os.path.join(recipe_steps_dir, f"{step}.py"), "w") as f:
            f.write(template.render())

    # Data directory (crea solo se inesistente)
    data_dir = os.path.join(recipe_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    # metadata directory (crea solo se inesistente)
    metadata_dir = os.path.join(recipe_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)

    # File profiles directory (crea solo se inesistente)
    profiles_dir = os.path.join(recipe_dir, "profiles")
    os.makedirs(profiles_dir, exist_ok=True)

    # File profiles directory
    for profile in ["local"]:
        template = env.get_template(f"profiles_{profile}.yaml.jinja")
        with open(os.path.join(profiles_dir, f"{profile}.yaml"), "w") as f:
            f.write(template.render())

    _logger.info(f"Recipe '{recipe_name}' creata con successo in '{recipe_dir}'")


if __name__ == "__main__":
    cli()