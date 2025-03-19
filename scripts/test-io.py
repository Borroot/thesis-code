import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Copy from input file to output file.")
parser.add_argument("input_file", type=str, help="The path to the input file")
parser.add_argument("output_file", type=str, help="The path to the output file")

# Parse arguments
args = parser.parse_args()

# Read the input file and write to the output file
try:
    with open(args.input_file, "r") as f:
        with open(args.output_file, "w") as g:
            g.write(f.read())
except FileNotFoundError:
    print(f"The file {args.input_file} does not exist.")
except Exception as e:
    print(f"An error occurred: {e}")
