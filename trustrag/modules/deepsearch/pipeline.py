import os
import sys


# Automatically find the TrustRAG root directory
def find_trustrag_directory():
    # Get the absolute path of the current file
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)

    # Start from current directory and walk up until we find TrustRAG directory
    test_dir = current_dir
    while test_dir and os.path.dirname(test_dir) != test_dir:  # Stop at filesystem root
        # Check if this directory is named TrustRAG
        if os.path.basename(test_dir) == "TrustRAG":
            return test_dir

        # Check if TrustRAG is a direct subdirectory
        potential_trustrag = os.path.join(test_dir, "TrustRAG")
        if os.path.exists(potential_trustrag) and os.path.isdir(potential_trustrag):
            return potential_trustrag

        # Move up one directory
        test_dir = os.path.dirname(test_dir)

    # If nothing is found, return None
    return None


# Find and add TrustRAG to path
trustrag_dir = find_trustrag_directory()
print(trustrag_dir)
if trustrag_dir:
    sys.path.append(trustrag_dir)
    print(f"Added TrustRAG directory to path: {trustrag_dir}")
else:
    print("Warning: Could not locate TrustRAG directory automatically.")


import asyncio
import typer
from functools import wraps
from prompt_toolkit import PromptSession
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

from trustrag.modules.deepsearch.action import deep_research, write_final_report
from trustrag.modules.deepsearch.feedback import generate_feedback
from trustrag.modules.deepsearch.agent.providers import AIClientFactory
from trustrag.modules.deepsearch.config import EnvironmentConfig



app = typer.Typer()
console = Console()
session = PromptSession()


def coro(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))

    return wrapper


async def async_prompt(message: str, default: str = "") -> str:
    """Async wrapper for prompt_toolkit."""
    return await session.prompt_async(message)


@app.command()
@coro
async def main(
    concurrency: int = typer.Option(
        default=2, help="Number of concurrent tasks, depending on your API rate limits."
    ),
):
    """Deep Research CLI"""
    console.print(
        Panel.fit(
            "[bold blue]Deep Research Assistant[/bold blue]\n"
            "[dim]An AI-powered research tool[/dim]"
        )
    )

    service = EnvironmentConfig.get_default_provider()

    console.print(f"🛠️ Using [bold green]{service.upper()}[/bold green] service.")

    client = AIClientFactory.get_client()

    # Get the model for the current provider
    model = AIClientFactory.get_model()

    # Get initial inputs with clear formatting
    query = await async_prompt("\n🔍 你想研究什么？")
    console.print()

    breadth_prompt = "📊 研究广度,查询扩展的个数（建议2-10）[4]："
    breadth = int((await async_prompt(breadth_prompt)) or "4")
    console.print()

    depth_prompt = "🔍研究深度，递归检索的深度（建议1-5）[2]："
    depth = int((await async_prompt(depth_prompt)) or "2")
    console.print()

    # # First show progress for research plan
    # console.print("\n[yellow]创建研究计划的链路...[/yellow]")
    # follow_up_questions = await generate_feedback(query, client, model)
    #
    # # Then collect answers separately from progress display
    # console.print("\n[bold yellow]Follow-up Questions:[/bold yellow]")
    # answers = []
    # for i, question in enumerate(follow_up_questions, 1):
    #     console.print(f"\n[bold blue]Q{i}:[/bold blue] {question}")
    #     answer = await async_prompt("➤ Your answer: ")
    #     answers.append(answer)
    #     console.print()

    # combined_query = f"""
    # 初始查询：{query}
    # 后续问题和答案：
    # {chr(10).join(f"Q: {q} A: {a}" for q, a in zip(follow_up_questions, answers))}
    # """

    # Now use Progress for the research phase
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Do research
        task = progress.add_task(
            "[yellow]正在探究你的主题...[/yellow]", total=None
        )
        research_results = await deep_research(
            query=query,
            breadth=breadth,
            depth=depth,
            concurrency=concurrency,
            client=client,
            model=model,
        )
        progress.remove_task(task)

        # Show learnings
        console.print("\n[yellow]Learnings:[/yellow]")
        for learning in research_results["learnings"]:
            rprint(f"• {learning}")

        # Generate report
        task = progress.add_task("Writing final report...", total=None)
        report = await write_final_report(
            # prompt=combined_query,
            prompt=query,
            learnings=research_results["learnings"],
            visited_urls=research_results["visited_urls"],
            client=client,
            model=model,
        )
        progress.remove_task(task)

        # Show results
        console.print("\n[bold green]Research Complete![/bold green]")
        console.print("\n[yellow]Final Report:[/yellow]")
        console.print(Panel(report, title="Research Report"))

        # Show sources
        console.print("\n[yellow]Sources:[/yellow]")
        for url in research_results["visited_urls"]:
            rprint(f"• {url}")

        # Save report
        console.print("\n[dim]Report  been saved to output.md[/dim]")

        with open("output.md", "w",encoding="utf-8") as f:
            f.write(report)

def run():
    """Synchronous entry point for the CLI tool."""
    asyncio.run(app())


if __name__ == "__main__":
    asyncio.run(app())