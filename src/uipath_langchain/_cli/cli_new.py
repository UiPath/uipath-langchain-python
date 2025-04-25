import os
import shutil

import click
from uipath._cli.middlewares import MiddlewareResult
from uipath._cli.spinner import Spinner


def generate_script(target_directory):
    template_script_path = os.path.join(
        os.path.dirname(__file__), "_templates/main.py.template"
    )
    target_path = os.path.join(target_directory, "main.py")

    shutil.copyfile(template_script_path, target_path)

    template_langgraph_json_path = os.path.join(
        os.path.dirname(__file__), "_templates/langgraph.json.template"
    )
    target_path = os.path.join(target_directory, "langgraph.json")
    shutil.copyfile(template_langgraph_json_path, target_path)


def generate_pyproject(target_directory, project_name):
    project_toml_path = os.path.join(target_directory, "pyproject.toml")
    toml_content = f"""[project]
                        name = "{project_name}"
                        version = "0.0.1"
                        description = "{project_name}"
                        authors = [{{ name = "John Doe", email = "john.doe@myemail.com" }}]
                        dependencies = [
                            "uipath-langchain>=0.0.95",
                            "langchain-anthropic>=0.3.8",
                        ]
                        requires-python = ">=3.10"
                    """

    with open(project_toml_path, "w") as f:
        f.write(toml_content)


def langgraph_new_middleware(name: str) -> MiddlewareResult:
    """Middleware to create demo langchain agent"""
    spinner = Spinner("Creating demo agent...")
    directory = os.getcwd()

    try:
        generate_script(directory)
        click.echo(click.style("✓ ", fg="green", bold=True) + "Created main.py file")
        click.echo(
            click.style("✓ ", fg="green", bold=True) + "Created langgraph.json file"
        )
        generate_pyproject(directory, name)
        click.echo(
            click.style("✓ ", fg="green", bold=True) + "Created pyproject.toml file"
        )
        os.system("uv sync")
        spinner.start()
        ctx = click.get_current_context()
        init_cmd = ctx.parent.command.get_command(ctx, "init")  # type: ignore
        ctx.invoke(init_cmd)
        spinner.stop()
        click.echo(
            click.style("✓ ", fg="green", bold=True) + " Agent created successfully."
        )
        return MiddlewareResult(
            should_continue=False,
            info_message="""Usage example: ` uipath run agent '{"topic": "UiPath"}' `""",
        )
    except Exception as e:
        spinner.stop()
        return MiddlewareResult(
            should_continue=False,
            error_message=f"❌ Error creating demo agent {str(e)}",
            should_include_stacktrace=True,
        )
