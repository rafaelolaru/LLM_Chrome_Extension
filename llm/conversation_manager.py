import threading
import time
from typing import Dict, List
import redis
import json

class ConversationHistoryManager:
    def __init__(self, redis_client, flush_interval=5):
        self.redis_client = redis_client
        self.flush_interval = flush_interval
        self.conversations_cache: Dict[str, List[Dict[str, str]]] = {}
        self.lock = threading.Lock()
        threading.Thread(target=self._start_periodic_flush, daemon=True).start()

    def add_message(self, username: str, role: str, content: str):
        with self.lock:
            if username not in self.conversations_cache:
                self.conversations_cache[username] = []
            self.conversations_cache[username].append({"role": role, "content": content})

    def _start_periodic_flush(self):
        while True:
            time.sleep(self.flush_interval)
            self._flush_conversations_to_redis()

    def _flush_conversations_to_redis(self):
        with self.lock:
            for username, messages in self.conversations_cache.items():
                if messages:
                    redis_key = f"user:{username}:conversations"
                    for message in messages:
                        self.redis_client.rpush(redis_key, json.dumps(message))  # Store messages as JSON strings
            self.conversations_cache.clear()

    def get_conversation_history(self, username: str) -> List[Dict[str, str]]:
        redis_key = f"user:{username}:conversations"
        messages = self.redis_client.lrange(redis_key, 0, -1)
        return [json.loads(message) for message in messages]  # Convert JSON strings back to dictionaries

    def clear_conversation_history(self, username: str):
        redis_key = f"user:{username}:conversations"
        self.redis_client.delete(redis_key)
        with self.lock:
            if username in self.conversations_cache:
                del self.conversations_cache[username]
