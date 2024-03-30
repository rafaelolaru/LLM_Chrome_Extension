# conversation_manager.py
import threading
import time
from typing import Dict, List
import redis

class ConversationHistoryManager:
    def __init__(self, redis_client, flush_interval=300):
        self.redis_client = redis_client
        self.flush_interval = flush_interval
        self.conversations_cache: Dict[str, List[str]] = {}
        self.lock = threading.Lock()
        threading.Thread(target=self._start_periodic_flush, daemon=True).start()

    def add_message(self, username: str, user_message: str, assistant_message: str):
        conversation_entry = f"User: {user_message} Assistant: {assistant_message}."
        with self.lock:
            if username not in self.conversations_cache:
                self.conversations_cache[username] = []
            self.conversations_cache[username].append(conversation_entry)

    def _start_periodic_flush(self):
        while True:
            time.sleep(self.flush_interval)
            self._flush_conversations_to_redis()

    def _flush_conversations_to_redis(self):
        with self.lock:
            for username, messages in self.conversations_cache.items():
                if messages:
                    redis_key = f"user:{username}:conversations"
                    self.redis_client.rpush(redis_key, *messages)
            self.conversations_cache.clear()

    def get_conversation_history(self, username: str) -> List[str]:
        redis_key = f"user:{username}:conversations"
        return self.redis_client.lrange(redis_key, 0, -1)
