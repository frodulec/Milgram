
from autogen import AssistantAgent, agentchat, Agent, GroupChat, GroupChatManager
from app.autogen_app.human_agent import LengthChecker
from app.autogen_app.metrics import compound_metrics
from app.data_models import Summary
import os
from app.config import AutogenConf
from typing_extensions import Annotated
from app.validation.results_validation import Validator
import re


def remove_word_count(text: str):
    for characters_word in ["Zeichen", "Mots"]:
        pattern = r"\s*\(.*" + characters_word + r":\s*\d+\)"
    cleaned_text = re.sub(pattern, "", text)
    return cleaned_text


def geval_tool(input_text: Annotated[str, "Text before summarization"],
               summary: Annotated[str, "Summary"]
               ) -> str:
    # the Wizard does not support calling fucntions, so the tool is not used currently
    validator = Validator()
    dummy_text = Summary(
        text_id=1,
        content_id=1,
        input_text='',
        output_text='',
        author='autogen',
        created_at=0,
        lang='en',
        form='',
        genre='')
    dummy_text.input_text = input_text
    dummy_text.output_text = summary
    result = validator.validate(dummy_text, compound_metrics)
    return str(result.reason)


def text_length_tool(text: str) -> str:
    return str(len(text))


def create_summary_w_autogen(
        summary: Summary,
        conf: AutogenConf
        ) -> Summary | None:


    if conf.model.startswith('gpt-'):
        llm_config = {"model": conf.model, "api_key": os.environ["OPENAI_API_KEY"]}
    else:
        llm_config = {"model": conf.model,
                    "api_key": os.environ["TOGETHER_API_KEY"],
                    "base_url": "https://api.together.xyz/v1"}

    conf_3_5 = {"model": "gpt-3.5-turbo-0125", "api_key": os.environ["OPENAI_API_KEY"]}

    critic_assistant = AssistantAgent(
        "Summarizer critic",
        llm_config=llm_config,
        system_message=conf.critic_instructions,
        is_termination_msg=lambda x: "TERMINATE" in x["content"] if x["content"] else False,)

    summarizer_assistant = AssistantAgent(
        "Summarizer",
        llm_config=llm_config,
        system_message=conf.summarizer_instructions,
        is_termination_msg=lambda x: "TERMINATE" in x["content"] if x["content"] else False,)
    
    text_length_checker = LengthChecker(
        "human_proxy",
        llm_config=False,  # no LLM used for human proxy
        human_input_mode="ALWAYS"
    )

    def group_chat_order(last_speaker: Agent, groupchat: GroupChat):
        """Critic to summarizer"""
        messages = groupchat.messages
        if len(messages) == 1:
            return summarizer_assistant
        if last_speaker is critic_assistant:
            return summarizer_assistant
        elif last_speaker is summarizer_assistant:
            return text_length_checker
        else:
            return critic_assistant


    if conf.call_functions:
        critic_assistant.register_for_llm(
            name="geval_tool",
            description="GEval score breakdown tool."
            )(geval_tool)
        summarizer_assistant.register_for_execution(
            name="geval_tool"
            )(geval_tool)
    
        agentchat.register_function(geval_tool,
                                    caller=critic_assistant,
                                    executor=summarizer_assistant,
                                    name="geval_tool",
                                    description='Get summary score breakdown for a given text.')
    
        agentchat.register_function(text_length_tool,
                                    caller=critic_assistant,
                                    executor=summarizer_assistant,
                                    name="text_length_tool",
                                    description='Get the length of a given text.')

    # chat = critic_assistant.initiate_chat(summarizer_assistant,
    #                         max_turns=10,
    #                         message=conf.initial_message + summary.input_text,
    #                         is_termination_msg=termination_check)

    group_chat = GroupChat(
        agents=[critic_assistant, summarizer_assistant, text_length_checker],
        messages=[],
        max_round=15*3,
        speaker_selection_method=group_chat_order,
    )
    manager = GroupChatManager(groupchat=group_chat, llm_config=conf_3_5)
    critic_assistant.initiate_chat(
        manager,
        message=conf.initial_message + summary.input_text)
    
    last_summary = None
    last_review = None
    chat_history = manager.chat_messages
    chat_items = chat_history.items()
    for agent_obj, messages in chat_items:
        if(agent_obj.name == "Summarizer"):
            summarizer_messages = [msg['content'] for msg in messages]
            last_review = summarizer_messages[-1]
            last_summary = summarizer_messages[-3]
        else:
            messages = [msg['content'] for msg in messages]

    agents = manager.groupchat.agents
    # last message should contain Terminate and
    # then the one before last should be perfect summary
    if not last_summary:
        return None

    # sometimes the summaizer adds the word count to the summary
    # whichj looks exaclty like this: (Zeichen: 123) which is at the end of the summary
    last_summary = remove_word_count(last_summary)

    costs_sum = sum([bot.get_total_usage()['total_cost'] for bot in agents])
    if 'terminate' in last_review.lower() or 'terminier' in last_review.lower():
        # print('Summary was approved')
        return Summary(
            text_id=summary.text_id,
            content_id=summary.content_id,
            input_text=summary.input_text,
            output_text=last_summary,
            author=summary.author,
            created_at=summary.created_at,
            lang=summary.lang,
            form=summary.form,
            genre=summary.genre,
            creation_cost=costs_sum,
            evaluation_cost=0,
            creation_model=conf.model)

    else:
        # print('Summary was not approved')
        return None