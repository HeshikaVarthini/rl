from cli.arg_parser import create_parser
from cli.commands import execute_command

def main():
    """Main function: Parses command line arguments and executes the appropriate subcommand"""
    # Create command line argument parser
    parser = create_parser()
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute command
    return execute_command(args)

if __name__ == "__main__":
    exit(main())