from autogen import Agent, GroupChat, ChatResult
from models import States, Roles
import json

# def group_chat_order(last_speaker: Agent, groupchat: GroupChat):
#     """Order of speaking.
#     1. Welcome of the professor.
#     2. Participant-operator asks a question.
#     3. Participant-LEARNER answers.
#     4. Professor verifies the answer. Tell the participant-operator if he should press the button"""
#     global CURRENT_STATE

#     messages = groupchat.messages
#     if CURRENT_STATE == States.START:
#         CURRENT_STATE = States.QUESTION
#         return proffesor
#     if CURRENT_STATE == States.QUESTION:
#         if last_speaker is participant_operator:
#             return LEARNER
#         elif last_speaker is LEARNER:
#             CURRENT_STATE = States.ANSWER_VERIFICATION
#             return proffesor
#         else:
#             return participant_operator

#     if CURRENT_STATE == "VERIFY":
#         return proffesor
#     if CURRENT_STATE == "BUTTON":
#         return button
#     if len(messages) == 1:
#         return proffesor
#     if last_speaker is proffesor:
#         return LEARNER
#     elif last_speaker is LEARNER:
#         return button
#     else:
#         return proffesor


def check_termination(message) -> bool:
    if message["content"]:
        return "goodbye" in message["content"].lower()
    return False


def convert_chat_history_to_json(
    chat: ChatResult, output_file_path: str = "conversation.json"
) -> list[dict]:
    agent_names_mapping = {
        Roles.PROFESSOR.value: "Professor",
        Roles.LEARNER.value: "Learner",
        Roles.PARTICIPANT.value: "Participant",
    }
    messages_of_people = [
        message
        for message in chat.chat_history
        if message["name"] in agent_names_mapping
    ]
    data = [
        {
            "speaker": agent_names_mapping[message["name"]],
            "text": "ELECTRIC_SHOCK_IMAGE"
            if any(
                tool_call["function"]["name"] == "Administer-shock"
                for tool_call in message.get("tool_calls", [])
            )
            else message["content"],
            "delay": len(message["content"]) / 30 + 1,  # 1 as minimum delay
        }
        for message in messages_of_people
    ]
    # dump
    with open(output_file_path, "w") as f:
        json.dump(data, f, indent=4)
    return data


def load_conversation_dictionary(file_path: str = "conversation.json") -> list[dict]:
    with open(file_path, "r") as f:
        data = json.load(f)
    return data
