import sys
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.table import Table

# Import our existing RAG backend (no web server needed!)
from app.rag.pipeline import query
from app.rag.retriever import get_retriever
from app.rag.embedder import get_embedder

console = Console()

def main():
    # Print welcome banner
    console.print(Panel.fit(
        "[bold cyan]BEL Technical Mentorship — Offline AI Interface[/bold cyan]\n"
        "Fault Code Lookup System (CLI Mode)\n"
        "Type [bold red]exit[/bold red] or [bold red]quit[/bold red] to leave.", 
        border_style="cyan"
    ))
    
    # Pre-load the models with a loading spinner
    with console.status("[bold yellow]Initializing offline AI core & loading FAISS index...[/bold yellow]", spinner="dots"):
        try:
            get_retriever()
            get_embedder()
        except Exception as e:
            console.print(f"[bold red]Startup Error:[/bold red] {e}")
            console.print("[yellow]Hint: Have you run ingestion yet? Run `python -m app.ingestion.ingest`[/yellow]")
            sys.exit(1)

    console.print("[bold green]System Ready. Running 100% locally.[/bold green]\n")

    while True:
        try:
            user_input = Prompt.ask("\n[bold blue]>[/bold blue] ")
            
            if user_input.lower().strip() in ("exit", "quit"):
                console.print("[dim]Shutting down...[/dim]")
                break
            if not user_input.strip():
                continue
                
            # Run the query with a status spinner
            with console.status("[bold magenta]Retrieving knowledge & generating response...[/bold magenta]", spinner="bouncingBar"):
                result = query(user_input)
                
            # If Ollama is down or threw an error (like OOM)
            if result.error:
                console.print(f"[bold red]Backend Error:[/bold red] {result.error}")
                console.print("[yellow]Make sure Ollama is running and has enough RAM/VRAM.[/yellow]")
                continue
                
            # Print the AI's answer nicely formatted as Markdown
            answer_md = Markdown(result.answer)
            console.print(Panel(answer_md, title="[bold green]AI Response[/bold green]", border_style="green", expand=False))
            
            # Print retrieved chunks as a debug/citation table if they exist
            if result.retrieved_chunks:
                table = Table(title="[dim]Retrieved Context (Traceability)[/dim]", show_header=True, header_style="bold dim")
                table.add_column("Error Code", style="cyan dim")
                table.add_column("Description", style="dim")
                table.add_column("Score", justify="right", style="dim")
                
                for chunk in result.retrieved_chunks:
                    table.add_row(
                        chunk.get("error_code", "N/A"),
                        chunk.get("error_description", "N/A"),
                        f"{chunk.get('score', 0):.2f}"
                    )
                console.print(table)

            if result.guardrail_triggered:
                console.print("[bold yellow]⚠ Note:[/bold yellow] Response was regenerated due to citation hallucination guardrail.")
                
            console.print(f"[dim]Latency: {result.latency_ms}ms[/dim]")
            
        except KeyboardInterrupt:
            console.print("\n[dim]Shutting down...[/dim]")
            break
        except Exception as e:
            console.print(f"\n[bold red]System Error:[/bold red] {e}")

if __name__ == "__main__":
    main()
