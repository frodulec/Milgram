from autogen import AssistantAgent


class ButtonNarrator(AssistantAgent):
    def __init__(self, name, system_message, llm_config):
        super().__init__(
            name=name,
            system_message=system_message,
            llm_config=llm_config,
            human_input_mode="ALWAYS",
        )

    def get_human_input(self, prompt: str) -> str:
        """Check if button was pressed. If it was pressed, print also the current voltage and the consequences of the button press."""
        return input("Your input: ")
