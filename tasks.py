# -*- encoding: utf-8 -*-
# created by aochujie
from pathlib import Path

from invoke import task

# from invoke.context import Context


@task()
def black(c):
    """执行 black 格式化命令"""
    print("=" * 5, "run black", "=" * 5)
    c.run("black ./src ./tests")
    print("=" * 15)


@task()
def isort(c):
    """执行 isort 命令"""
    print("=" * 5, "run isort", "=" * 5)
    c.run("isort -y --recursive ./src ./tests")
    print("=" * 15)


@task()
def flake(c):
    """执行 flake8 代码检查"""
    print("=" * 5, "run flake", "=" * 5)
    c.run("flake8 ./src ./tests")
    print("=" * 15)


@task()
def pylint(c):
    """执行 flake8 代码检查"""
    print("=" * 5, "run pylint", "=" * 5)
    c.run("pylint ./src --exit-zero")
    print("=" * 15)


@task(black, isort, flake, pylint, default=True)
def check(c):
    print("~" * 5, "check finish!", "~" * 5)


@task()
def doc(c, no_browser=False):
    """构建 sphinx-doc """
    print("###### run doc")
    with c.cd("docs"):
        c.run("make html")
    if not no_browser:
        import webbrowser

        index = Path(__file__).parent / "docs" / "_build" / "html" / "index.html"
        index.absolute()
        webbrowser.open(f"file:///{index.absolute()}")
