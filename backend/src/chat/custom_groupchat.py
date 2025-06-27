from autogen.agentchat.groupchat import GroupChat
from typing import Any, Optional, Union
from autogen.io.base import IOStream
from autogen.events.agent_events import (
    ClearAgentsHistoryEvent,
    GroupChatResumeEvent,
    GroupChatRunChatEvent,
    SelectSpeakerEvent,
    SelectSpeakerInvalidInputEvent,
    SelectSpeakerTryCountExceededEvent,
    SpeakerAttemptFailedMultipleAgentsEvent,
    SpeakerAttemptFailedNoAgentsEvent,
    SpeakerAttemptSuccessfulEvent,
    TerminationEvent,
)

def extract_from_tags(
    text: str,
    tag: str,
) -> str:
    """Extracts a value from tags in the text.
    for example <tag>value</tag>"""
    tags = text.split(f"<{tag}>")
    if len(tags) > 1:
        tag_value = tags[1].split(f"</{tag}>")[0]
        return tag_value.strip()
    else:
        return ""

class CustomGroupChat(GroupChat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.select_speaker_prompt_template = "Read the above conversation. Then select the next role from {agentlist} to play. Write one sentance of reasoning and then return the role in tags, for example: <speaker>agent_name</speaker>"

    def _validate_speaker_name(
        self, recipient, messages, sender, config, attempts_left, attempt, agents
    ) -> tuple[bool, Optional[Union[str, dict[str, Any]]]]:
        """Validates the speaker response for each round in the internal 2-agent
        chat within the  auto select speaker method.

        Used by auto_select_speaker and a_auto_select_speaker.
        """
        # Validate the speaker name selected
        # extract from <speaker> in the message content
        select_name = extract_from_tags(messages[-1]["content"].strip(), "speaker")

        mentions = self._mentioned_agents(select_name, agents)

        # Output the query and requery results
        if self.select_speaker_auto_verbose:
            iostream = IOStream.get_default()
            no_of_mentions = len(mentions)
            if no_of_mentions == 1:
                # Success on retry, we have just one name mentioned
                iostream.send(
                    SpeakerAttemptSuccessfulEvent(
                        mentions=mentions,
                        attempt=attempt,
                        attempts_left=attempts_left,
                        select_speaker_auto_verbose=self.select_speaker_auto_verbose,
                    )
                )
            elif no_of_mentions == 1:
                iostream.send(
                    SpeakerAttemptFailedMultipleAgentsEvent(
                        mentions=mentions,
                        attempt=attempt,
                        attempts_left=attempts_left,
                        select_speaker_auto_verbose=self.select_speaker_auto_verbose,
                    )
                )
            else:
                iostream.send(
                    SpeakerAttemptFailedNoAgentsEvent(
                        mentions=mentions,
                        attempt=attempt,
                        attempts_left=attempts_left,
                        select_speaker_auto_verbose=self.select_speaker_auto_verbose,
                    )
                )

        if len(mentions) == 1:
            # Success on retry, we have just one name mentioned
            selected_agent_name = next(iter(mentions))

            # Add the selected agent to the response so we can return it
            messages.append({"role": "user", "content": f"[AGENT SELECTED]{selected_agent_name}"})

        elif len(mentions) > 1:
            # More than one name on requery so add additional reminder prompt for next retry

            if attempts_left:
                # Message to return to the chat for the next attempt
                agentlist = f"{[agent.name for agent in agents]}"

                return True, {
                    "content": self.select_speaker_auto_multiple_template.format(agentlist=agentlist),
                    "name": "checking_agent",
                    "override_role": self.role_for_select_speaker_messages,
                }
            else:
                # Final failure, no attempts left
                messages.append({
                    "role": "user",
                    "content": f"[AGENT SELECTION FAILED]Select speaker attempt #{attempt} of {attempt + attempts_left} failed as it returned multiple names.",
                })

        else:
            # No names at all on requery so add additional reminder prompt for next retry

            if attempts_left:
                # Message to return to the chat for the next attempt
                agentlist = f"{[agent.name for agent in agents]}"

                return True, {
                    "content": self.select_speaker_auto_none_template.format(agentlist=agentlist),
                    "name": "checking_agent",
                    "override_role": self.role_for_select_speaker_messages,
                }
            else:
                # Final failure, no attempts left
                messages.append({
                    "role": "user",
                    "content": f"[AGENT SELECTION FAILED]Select speaker attempt #{attempt} of {attempt + attempts_left} failed as it did not include any agent names.",
                })

        return True, None