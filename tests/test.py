from colorama import Fore, Style

def print_success(message):
    print(f"{Fore.GREEN}[+]{Style.RESET_ALL} {message}")

# Example usage:
print_success("Installation complete!")
print_success("Process successful!")
