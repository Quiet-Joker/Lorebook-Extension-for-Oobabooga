import threading
from collections import deque


class LorebookState:
    def __init__(self):
        self.state_lock = threading.RLock()
        self.current_lorebook: dict | None = None
        self.next_uid: int = 1
        self.active_lorebooks: dict = {}

        self.active_entries_cache: list = []
        self.active_entries_version: int = 0
        self.active_entries_cache_version: int = -1

        self.cur_injected_lock = threading.Lock()
        self.cur_injected: set = set()

        self.last_injection_lock = threading.Lock()
        self.last_injection_info: dict = {"entries": [], "interrupts": 0, "total_tokens": 0}

        self.last_notebook_injection_lock = threading.Lock()
        self.last_notebook_injection_info: dict = {"entries": [], "interrupts": 0, "total_tokens": 0}

        self.injection_history_lock = threading.Lock()
        self.injection_history: deque = deque(maxlen=60)
        self.injection_history_notebook: deque = deque(maxlen=60)

        self.label_tracking_lock = threading.Lock()
        self.prev_chat_labels: set = set()
        self.prev_notebook_labels: set = set()
        self.all_chat_labels: set = set()
        self.all_notebook_labels: set = set()
        self.prev_chat_eid_labels: dict = {}
        self.prev_notebook_eid_labels: dict = {}
        self.chat_turn_counter: int = 0
        self.notebook_turn_counter: int = 0


_st = LorebookState()


def _bump_active_version():
    _st.active_entries_version += 1
