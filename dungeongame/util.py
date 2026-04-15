def ask_for_input(prompt, options):
    """
    Ask for input from the user, and validate it against a list of options.
    """
    while True:
        response = input(prompt + "\n> ")
        if response.upper() in options:
            return response
        else:
            print("Invalid input. Please try again.")