from __future__ import annotations

import hashlib
import json
import logging

import openai

logger = logging.getLogger(__name__)


class AI:
    def __init__(self, model="gpt-4", temperature=0.0, cache=None):
        self.temperature = temperature
        self.cachedb = cache

        try:
            openai.Model.retrieve(model)
            self.model = model
        except openai.InvalidRequestError:
            print(
                f"Model {model} not available for provided API key. Reverting "
                "to gpt-3.5-turbo. Sign up for the GPT-4 wait list here: "
                "https://openai.com/waitlist/gpt-4-api"
            )
            self.model = "gpt-3.5-turbo"

    def start(self, system, user):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        return self.next(messages)

    def fsystem(self, msg):
        return {"role": "system", "content": msg}

    def fuser(self, msg):
        return {"role": "user", "content": msg}

    def fassistant(self, msg):
        return {"role": "assistant", "content": msg}

    # caching results from server (assuming temperature = 0)
    def hash_messages(self, messages):
        h = hashlib.sha256()
        for m in messages:
            h.update(str(m).encode())
        # include the model/source in the hashkey
        h.update(str(self.model).encode())
        return h.hexdigest()

    def from_cache(self, messages):
        results = None
        h = self.hash_messages(messages)
        try:
            results = self.cachedb[h]
        except Exception:
            pass
        # combine messages into hash
        return results

    def to_cache(self, messages, results):
        h = self.hash_messages(messages)
        json_dump = json.dumps(messages, indent=2)
        self.cachedb[h + ".prompt"] = json_dump
        self.cachedb[h] = results

    def next(self, messages: list[dict[str, str]], prompt=None, cache=None):
        if prompt:
            messages += [{"role": "user", "content": prompt}]

        content = None
        if self.cachedb:
            content = self.from_cache(messages)

        if not content:
            logger.debug(f"Creating a new chat completion: {messages}")
            response = openai.ChatCompletion.create(
                messages=messages,
                stream=True,
                model=self.model,
                temperature=self.temperature,
            )

            chat = []
            for chunk in response:
                delta = chunk["choices"][0]["delta"]
                msg = delta.get("content", "")
                print(msg, end="")
                chat.append(msg)

            content = "".join(chat)

            if self.cachedb:
                chat = self.to_cache(messages, content)

        print()
        messages += [{"role": "assistant", "content": content}]
        logger.debug(f"Chat completion finished: {messages}")
        return messages
