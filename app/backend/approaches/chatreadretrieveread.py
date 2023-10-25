from typing import Any, AsyncGenerator

import openai
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import QueryType

from core.messagebuilder import MessageBuilder
from core.modelhelper import get_token_limit
from text import nonewlines


class ChatReadRetrieveReadApproach:
    # Chat roles
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

    """
    Simple retrieve-then-read implementation, using the Cognitive Search and OpenAI APIs directly. It first retrieves
    top documents from search, then constructs a prompt with them, and then uses OpenAI to generate an completion
    (answer) with that prompt.
    """
    system_message_chat_conversation = """You are a digital assistant trained to assist with questions related to the diagnostic viewer product and its installation. How would you assist company employees facing issues or having questions about these topics? Remember to provide succinct answers, exclusively based on facts provided in the sources below.Guidelines:
Only use facts from the listed sources to answer questions. If the information isn't available in the sources, state that you don't know.
If a clarifying question could make your answer more precise, feel free to ask the user.
Answer in the language the question was asked if it's not in English.
Present tabular information as an HTML table, avoiding the use of markdown format.
Source Citation:
Always include the source name for each fact you use in the response. Use square brackets for citation, e.g., [info1.txt]. Do not amalgamate multiple sources; list each one separately, e.g., [info1.txt][info2.pdf].
{follow_up_questions_prompt}
{injected_prompt}
"""
    follow_up_questions_prompt_content = """Generate three very brief follow-up questions that the user would likely ask next about their GE healthcare Imaging product and diagnostic viewer.
Use double angle brackets to reference the questions, e.g. <<Are there exclusions for prescriptions?>>.
Try not to repeat questions that have already been asked.
Only generate questions and do not generate any text before or after the questions, such as 'Next Questions'"""

    query_prompt_template = """Below is a history of the conversation so far, and a new question asked by the user that needs to be answered by searching in a knowledge base about AW product features and server installation and  and the employee handbook.
Generate a search query based on the conversation and the new question.
Do not include cited source filenames and document names e.g info.txt or doc.pdf in the search query terms.
Do not include any text inside [] or <<>> in the search query terms.
Do not include any special characters like '+'.
If the question is not in English, translate the question to English before generating the search query.
If you cannot generate a search query, return just the number 0.
"""
    query_prompt_few_shots = [
        {'role' : USER, 'content' : 'What is AW Server?' },
        {'role' : ASSISTANT, 'content' : 'Show AW Server - Product Description' },
        {'role' : USER, 'content' : 'What are AW server system components?' },
        {'role' : ASSISTANT, 'content' : 'The three basic components in the AW Server system are:1. The AW SERVER. This is the hardware foundation for the AW SERVER system, and is the direct responsibility of GEHC, including repair and/or replacement by vendors (HP) acting as agents of GEHC. 2. The NETWORK, which connects the AW Server to the clients, PACS, etc. The network is NOT the responsibility of GEHC. 3. The CLIENTS, which are provided by the customer. Other than installing client software on ONE customer PC, the clients are NOT the responsibility of GEHC.' }
    ]

    def __init__(self, search_client: SearchClient, chatgpt_deployment: str, chatgpt_model: str, embedding_deployment: str, sourcepage_field: str, content_field: str):
        self.search_client = search_client
        self.chatgpt_deployment = chatgpt_deployment
        self.chatgpt_model = chatgpt_model
        self.embedding_deployment = embedding_deployment
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field
        self.chatgpt_token_limit = get_token_limit(chatgpt_model)

    async def run_until_final_call(self, history: list[dict[str, str]], overrides: dict[str, Any], should_stream: bool = False) -> tuple:
        has_text = overrides.get("retrieval_mode") in ["text", "hybrid", None]
        has_vector = overrides.get("retrieval_mode") in ["vectors", "hybrid", None]
        use_semantic_captions = True if overrides.get("semantic_captions") and has_text else False
        top = overrides.get("top") or 3
        exclude_category = overrides.get("exclude_category") or None
        filter = "category ne '{}'".format(exclude_category.replace("'", "''")) if exclude_category else None

        user_q = 'Generate search query for: ' + history[-1]["user"]

        # STEP 1: Generate an optimized keyword search query based on the chat history and the last question
        messages = self.get_messages_from_history(
            self.query_prompt_template,
            self.chatgpt_model,
            history,
            user_q,
            self.query_prompt_few_shots,
            self.chatgpt_token_limit - len(user_q)
            )

        chat_completion = await openai.ChatCompletion.acreate(
            deployment_id=self.chatgpt_deployment,
            model=self.chatgpt_model,
            messages=messages,
            temperature=0.0,
            max_tokens=32,
            n=1)

        query_text = chat_completion.choices[0].message.content
        if query_text.strip() == "0":
            query_text = history[-1]["user"] # Use the last user input if we failed to generate a better query

        # STEP 2: Retrieve relevant documents from the search index with the GPT optimized query

        # If retrieval mode includes vectors, compute an embedding for the query
        if has_vector:
            query_vector = (await openai.Embedding.acreate(engine=self.embedding_deployment, input=query_text))["data"][0]["embedding"]
        else:
            query_vector = None

         # Only keep the text query if the retrieval mode uses text, otherwise drop it
        if not has_text:
            query_text = None

        # Use semantic L2 reranker if requested and if retrieval mode is text or hybrid (vectors + text)
        if overrides.get("semantic_ranker") and has_text:
            r = await self.search_client.search(query_text,
                                          filter=filter,
                                          query_type=QueryType.SEMANTIC,
                                          query_language="en-us",
                                          query_speller="lexicon",
                                          semantic_configuration_name="default",
                                          top=top,
                                          query_caption="extractive|highlight-false" if use_semantic_captions else None,
                                          vector=query_vector,
                                          top_k=50 if query_vector else None,
                                          vector_fields="embedding" if query_vector else None)
        else:
            r = await self.search_client.search(query_text,
                                          filter=filter,
                                          top=top,
                                          vector=query_vector,
                                          top_k=50 if query_vector else None,
                                          vector_fields="embedding" if query_vector else None)
        if use_semantic_captions:
            results = [doc[self.sourcepage_field] + ": " + nonewlines(" . ".join([c.text for c in doc['@search.captions']])) async for doc in r]
        else:
            results = [doc[self.sourcepage_field] + ": " + nonewlines(doc[self.content_field]) async for doc in r]
        content = "\n".join(results)

        follow_up_questions_prompt = self.follow_up_questions_prompt_content if overrides.get("suggest_followup_questions") else ""

        # STEP 3: Generate a contextual and content specific answer using the search results and chat history

        # Allow client to replace the entire prompt, or to inject into the exiting prompt using >>>
        prompt_override = overrides.get("prompt_template")
        if prompt_override is None:
            system_message = self.system_message_chat_conversation.format(injected_prompt="", follow_up_questions_prompt=follow_up_questions_prompt)
        elif prompt_override.startswith(">>>"):
            system_message = self.system_message_chat_conversation.format(injected_prompt=prompt_override[3:] + "\n", follow_up_questions_prompt=follow_up_questions_prompt)
        else:
            system_message = prompt_override.format(follow_up_questions_prompt=follow_up_questions_prompt)

        messages = self.get_messages_from_history(
            system_message,
            self.chatgpt_model,
            history,
            history[-1]["user"]+ "\n\nSources:\n" + content, # Model does not handle lengthy system messages well. Moving sources to latest user conversation to solve follow up questions prompt.
            max_tokens=self.chatgpt_token_limit)
        msg_to_display = '\n\n'.join([str(message) for message in messages])

        extra_info = {"data_points": results, "thoughts": f"Searched for:<br>{query_text}<br><br>Conversations:<br>" + msg_to_display.replace('\n', '<br>')}
        chat_coroutine = openai.ChatCompletion.acreate(
                deployment_id=self.chatgpt_deployment,
                model=self.chatgpt_model,
                messages=messages,
                temperature=overrides.get("temperature") or 0.7,
                max_tokens=1024,
                n=1,
                stream=should_stream)
        return (extra_info, chat_coroutine)

    async def run_without_streaming(self, history: list[dict[str, str]], overrides: dict[str, Any]) -> dict[str, Any]:
        extra_info, chat_coroutine = await self.run_until_final_call(history, overrides, should_stream=False)
        chat_content = (await chat_coroutine).choices[0].message.content
        extra_info["answer"] = chat_content
        return extra_info

    async def run_with_streaming(self, history: list[dict[str, str]], overrides: dict[str, Any]) -> AsyncGenerator[dict, None]:
        extra_info, chat_coroutine = await self.run_until_final_call(history, overrides, should_stream=True)
        yield extra_info
        async for event in await chat_coroutine:
            yield event


    def get_messages_from_history(self, system_prompt: str, model_id: str, history: list[dict[str, str]], user_conv: str, few_shots = [], max_tokens: int = 4096) -> list:
        message_builder = MessageBuilder(system_prompt, model_id)

        # Add examples to show the chat what responses we want. It will try to mimic any responses and make sure they match the rules laid out in the system message.
        for shot in few_shots:
            message_builder.append_message(shot.get('role'), shot.get('content'))

        user_content = user_conv
        append_index = len(few_shots) + 1

        message_builder.append_message(self.USER, user_content, index=append_index)

        for h in reversed(history[:-1]):
            if bot_msg := h.get("bot"):
                message_builder.append_message(self.ASSISTANT, bot_msg, index=append_index)
            if user_msg := h.get("user"):
                message_builder.append_message(self.USER, user_msg, index=append_index)
            if message_builder.token_length > max_tokens:
                break

        messages = message_builder.messages
        return messages
