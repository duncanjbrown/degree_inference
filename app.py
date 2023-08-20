import click
from degree_inference.openai import OpenAIClient
from degree_inference.bigquery import BigQueryClient

@click.group()
def cli():
    pass

@cli.command(help="Send unstructured data to GPT-4")
def gpt():
    from degree_inference.gpt import GPT

    gpt = GPT(OpenAIClient(), BigQueryClient())
    gpt.infer(n=10)

@cli.command(help="Run inference")
def infer():
    from cli import inference

    inference.run()

if __name__ == "__main__":
    cli()
