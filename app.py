import click
from degree_inference.openai import OpenAIClient
from degree_inference.bigquery import BigQueryClient

@click.group()
def cli():
    pass

@cli.command(help="Send unstructured data to GPT-4")
@click.option('--count', help='Number of degrees to infer.', required=True)
@click.option('--offset', help='Number of degrees to offset before beginning inference. Useful for resuming when something goes wrong.')
def gpt(count, offset):
    from degree_inference.gpt import GPT

    gpt = GPT(OpenAIClient(), BigQueryClient())
    gpt.infer(n=int(count),offset=int(offset))

@cli.command(help="Run inference")
@click.option('--model', help='The path to the model', required=True)
@click.option('--input', help='The path to the inputs as a newline-separated file', required=True)
def infer(model, input):
    from degree_inference import inference

    f = open(input, "r")
    xs = f.readlines()

    print(inference.run(model,xs))

if __name__ == "__main__":
    cli()
